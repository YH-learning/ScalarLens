"""Exact numeric checks, not benchmark-quality claims from synthetic data."""
import copy
import json
from pathlib import Path
import pytest
import torch

from scalarlens import ScalarLensEmbedding, StableCoordinates
from scalarlens._legacy.fast_coupled_embedding import FastCoupledEmbedding
from scalarlens._legacy.numerical_embeddings import DEEREmbedding
from scalarlens.results import aggregate, read_csv, reproduce, summarize_new_runs

ROOT = Path(__file__).resolve().parents[1]


def pair():
    edges = torch.linspace(-3, 3, 17).repeat(3, 1)
    torch.manual_seed(2026)
    old = FastCoupledEmbedding(DEEREmbedding(edges, 16), ['N1', 'N2', 'N3'], 16, {'C1': 16, 'C2': 16})
    torch.manual_seed(2026)
    new = ScalarLensEmbedding(StableCoordinates(edges, 16), ['N1', 'N2', 'N3'], 16, {'C1': 16, 'C2': 16})
    return old, new


def inputs():
    torch.manual_seed(123)
    return torch.randn(12, 3), {c: torch.randn(12, 16) for c in ('C1', 'C2')}


def test_exact_state_forward_backward_and_adam():
    old, new = pair()
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
        else:
            assert value == new.state_dict()[key]
    x, context = inputs()
    optimizers = [torch.optim.Adam(model.parameters(), lr=.001) for model in (old, new)]
    for _ in range(3):
        predictions = [model(x, context) for model in (old, new)]
        torch.testing.assert_close(*predictions, atol=0, rtol=0)
        for model, optimizer, prediction in zip((old, new), optimizers, predictions):
            optimizer.zero_grad()
            prediction.square().mean().backward()
        for (_, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        for optimizer in optimizers:
            optimizer.step()


def test_context_and_checkpoint_invariants():
    _, model = pair()
    x, context = inputs()
    original = x.clone()
    local = model.local_encoder(x)
    y = model(x, context)
    shuffled = {key: value.flip(0) for key, value in context.items()}
    assert not torch.equal(y, model(x, shuffled))
    torch.testing.assert_close(local, model.local_encoder(x), atol=0, rtol=0)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert y.shape == (12, 3, 16)
    replica = copy.deepcopy(model)
    replica.load_state_dict(model.state_dict())
    torch.testing.assert_close(y, replica(x, context), atol=0, rtol=0)
    boundaries = model.local_encoder.current_boundaries()
    assert (boundaries.diff(dim=-1) > 0).all()


def test_cpu_auto_equivalence():
    _, model = pair()
    x, context = inputs()
    reference = model(x, context)
    model.set_execution_mode('auto')
    torch.testing.assert_close(reference, model(x, context), atol=0, rtol=0)


def test_registry_builder_preserves_initialization_and_rng():
    from types import SimpleNamespace
    from collections import OrderedDict
    from scalarlens._legacy.numerical_embeddings import build_numerical_embedding
    from scalarlens.factory import build_scalarlens
    from scalarlens.protocol import training_params
    spec = OrderedDict([(f'N{i}', dict(type='numeric', numeric_bin_edges=torch.linspace(-3, 3, 17).tolist()))
                        for i in (1, 2, 3)] + [(name, dict(type='categorical')) for name in ('C1', 'C2')])
    feature_map = SimpleNamespace(features=spec)
    params = training_params('DCNv2')['numeric_embedding_params']
    torch.manual_seed(2026)
    old = build_numerical_embedding('deer_scattering_fast', ['N1', 'N2', 'N3'], feature_map, 16, params)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(2026)
    new = build_scalarlens(['N1', 'N2', 'N3'], feature_map, 16, params)
    assert torch.equal(old_rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
        else:
            assert value == new.state_dict()[key]
    x, context = inputs()
    torch.testing.assert_close(old(x, context), new(x, context), atol=0, rtol=0)


def test_lr_uses_shared_full_dimensional_embedding(tmp_path):
    from collections import OrderedDict
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.layers.blocks.logistic_regression import LogisticRegression
    feature_map = FeatureMap('LRSmoke', str(tmp_path))
    feature_map.features = OrderedDict([('N1', dict(type='numeric', source='')),
                                       ('N2', dict(type='numeric', source=''))])
    feature_map.num_fields = 2
    feature_map.default_emb_dim = 16
    layer = LogisticRegression(feature_map, numeric_embedding_dim=16)
    with torch.no_grad():
        layer.numeric_weight.fill_(1.)
    tokens = torch.randn(8, 2, 16, requires_grad=True)
    # Raw values cannot influence the numeric LR branch when shared tokens are supplied.
    first = layer({'N1': torch.zeros(8), 'N2': torch.zeros(8)}, tokens)
    second = layer({'N1': torch.full((8,), 1e9), 'N2': torch.full((8,), -1e9)}, tokens)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first, tokens.sum((1, 2)).unsqueeze(1), atol=1e-5, rtol=1e-5)
    first.sum().backward()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


def test_recompute_and_full_table_audit(tmp_path):
    report = reproduce(ROOT/'reproduction/paper', tmp_path)
    assert report['status'] == 'PASS'
    assert report['per_seed_records'] == 1539
    assert report['scalarlens']['wins'] == 25


def test_duplicate_and_missing_records_fail():
    rows = read_csv(ROOT/'reproduction/paper/per_seed_1539.csv')
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(rows + [rows[0]])
    with pytest.raises(ValueError, match='task matrix'):
        aggregate(rows[1:])


def test_new_training_summary_uses_correct_archived_reference(tmp_path):
    rows = read_csv(ROOT/'reproduction/paper/per_seed_1539.csv')
    selected = [r for r in rows if r['dataset']=='AutoML_A' and r['backbone']=='DCNv2' and r['method']=='Fast']
    for i, row in enumerate(selected):
        task = tmp_path/'runs'/str(i)
        task.mkdir(parents=True)
        row['method'] = 'scalarlens'
        (task/'result.json').write_text(json.dumps(row))
    report = summarize_new_runs(tmp_path/'runs', tmp_path/'summary', ROOT/'reproduction/paper')
    assert report['tasks'] == 3
    summary = read_csv(tmp_path/'summary/summary.csv')[0]
    assert float(summary['test_auc_delta_vs_paper']) == 0.
    assert float(summary['test_logloss_delta_vs_paper']) == 0.


def test_timing_blocks_count_scope_without_gpu():
    from scalarlens.benchmark import timed_blocks
    calls = []
    syncs = []
    result = timed_blocks(iter(range(20)), calls.append, lambda: syncs.append(True),
                          warmup=2, steps=5, block_size=2)
    assert calls == list(range(7))
    assert len(syncs) == 4
    assert len(result['block_means_ms']) == 3


def test_imported_source_checksums():
    import hashlib
    for entry in json.loads((ROOT/'reproduction/metadata/source_imports.json').read_text()):
        assert hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest() == entry['sha256'], entry['file']


@pytest.mark.parametrize('backbone', ['DNN', 'DeepFM', 'WideDeep', 'NFM', 'PNN', 'DCN', 'DCNv2', 'AutoInt', 'FiBiNET'])
def test_backbone_forward_backward(backbone, tmp_path):
    from collections import OrderedDict
    from fuxictr.features import FeatureMap
    from scalarlens.backbones import get_backbone
    from scalarlens.protocol import training_params
    torch.manual_seed(2026)
    feature_map = FeatureMap('Smoke', str(tmp_path))
    feature_map.features = OrderedDict(
        [(f'N{i}', dict(type='numeric', source='', numeric_bin_edges=torch.linspace(-3, 3, 17).tolist()))
         for i in (1, 2, 3)] +
        [(f'C{i}', dict(type='categorical', source='', vocab_size=7, padding_idx=0)) for i in (1, 2)])
    feature_map.num_fields = 5
    feature_map.labels = ['Label']
    feature_map.default_emb_dim = 16
    feature_map.numeric_embedding = 'scalarlens'
    feature_map.numeric_embedding_params = training_params(backbone)['numeric_embedding_params']
    feature_map.numeric_embedding_plugins = ['scalarlens.plugin']
    feature_map.set_column_index()
    params = training_params(backbone)
    params.update(model_root=str(tmp_path), gpu=-1)
    model = get_backbone(backbone)(feature_map, **params)
    x, context = inputs()
    batch = {f'N{i+1}': x[:, i] for i in range(3)}
    batch.update({key: torch.arange(12) % 6 + 1 for key in context})
    batch['Label'] = torch.arange(12).float() % 2
    prediction = model(batch)['y_pred']
    assert prediction.shape == (12, 1)
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
