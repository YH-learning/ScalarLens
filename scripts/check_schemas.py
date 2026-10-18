"""Check old/new model identity on archived real-data schemas, without dataset rows."""
import copy
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
from fuxictr.features import FeatureMap
from scalarlens.backbones import BACKBONES, get_backbone
from scalarlens.protocol import training_params


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root/'outputs/validation/schema_equivalence'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((root/'reproduction/metadata/feature_maps').glob('*/feature_map.json')):
        metadata = json.loads(path.read_text())
        for backbone in BACKBONES:
            params = training_params(backbone)
            params.update(model_root=str(destination), gpu=-1)
            # Compare reference operations here; CPU regression separately
            # checks the frozen automatic engineering path in real training.
            params['numeric_embedding_params']['execution_mode'] = 'reference'
            params['numeric_embedding_params']['deer_execution_mode'] = 'reference'
            feature_map = FeatureMap(metadata['dataset_id'], str(path.parent))
            feature_map.load(str(path), params)
            torch.manual_seed(2026)
            public = get_backbone(backbone)(feature_map, **params)
            # Same consumer and frozen framework; only the encoder builder changes.
            old_map = copy.deepcopy(feature_map)
            old_map.numeric_embedding = 'deer_scattering_fast'
            old_params = copy.deepcopy(params)
            old_params['numeric_embedding'] = 'deer_scattering_fast'
            torch.manual_seed(2026)
            frozen = get_backbone(backbone)(old_map, **old_params)
            a, b = public.state_dict(), frozen.state_dict()
            assert a.keys() == b.keys()
            maximum = 0.
            tensor_count = 0
            for key, value in a.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, b[key], atol=0, rtol=0)
                    tensor_count += 1
                    if value.numel():
                        maximum = max(maximum, float((value.double()-b[key].double()).abs().max()))
                else:
                    assert value == b[key]
            # Artificial rows valid for each archived schema; not real examples.
            batch = {}
            for name, spec in feature_map.features.items():
                if spec['type'] == 'numeric':
                    edges = spec['numeric_bin_edges']
                    batch[name] = torch.linspace(float(edges[0]), float(edges[-1]), 8)
                elif spec['type'] == 'categorical':
                    batch[name] = torch.arange(8) % spec['vocab_size']
            for name in feature_map.labels:
                batch[name] = torch.arange(8).float() % 2
            public.eval()
            frozen.eval()
            with torch.no_grad():
                new_y, old_y = public(batch)['y_pred'], frozen(batch)['y_pred']
            torch.testing.assert_close(new_y, old_y, atol=0, rtol=0)
            records.append(dict(dataset=metadata['dataset_id'], backbone=backbone,
                                parameter_count=sum(p.numel() for p in public.parameters()),
                                tensors_checked=tensor_count, maximum_parameter_difference=maximum,
                                maximum_prediction_difference=float((new_y-old_y).abs().max())))
            del public, frozen
    report = dict(status='PASS', combinations=len(records),
                  scope='Archived real dataset schemas; artificial valid input rows, not real-data training.',
                  records=records)
    (root/'outputs/validation/real_schema_equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], combinations=len(records))))


if __name__ == '__main__':
    main()
