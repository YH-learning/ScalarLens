"""Validate prepared Parquet datasets or create an explicitly synthetic smoke fixture."""
import argparse
import hashlib
import json
from pathlib import Path


def make_smoke(output, train_rows=1024):
    import numpy as np
    import pandas as pd
    output = Path(output)
    if (output / 'feature_map.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    frames = {}
    for split, n in [('train', train_rows), ('valid', 256), ('test', 256)]:
        x = rng.normal(size=(n, 3)).astype('float32')
        c = rng.integers(1, 7, size=(n, 2))
        logits = x[:, 0] * np.where(c[:, 0] < 4, 1., -1.) + .3*x[:, 1] - .2*x[:, 2]
        y = rng.binomial(1, 1/(1+np.exp(-logits)))
        frames[split] = pd.DataFrame(dict(N1=x[:, 0], N2=x[:, 1], N3=x[:, 2],
                                          C1=c[:, 0], C2=c[:, 1], Label=y))
        frames[split].to_parquet(output / (split + '.parquet'), index=False)
    features = []
    for name in ('N1', 'N2', 'N3'):
        x = frames['train'][name].to_numpy()
        features.append({name: dict(type='numeric', source='', numeric_transform='identity',
                                   numeric_bin_edges=np.quantile(x, np.linspace(0, 1, 17)).tolist(),
                                   numeric_mean=float(x.mean()), numeric_std=float(x.std()))})
    features += [{name: dict(type='categorical', source='', vocab_size=7, padding_idx=0)} for name in ('C1', 'C2')]
    metadata = dict(dataset_id='ScalarLensSmoke', labels=['Label'], num_fields=5,
                    input_length=5, total_features=17, features=features, synthetic=True)
    (output / 'feature_map.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return validate(output)


def validate(data, sample_rows=8192):
    import numpy as np
    import pyarrow.parquet as pq
    data = Path(data)
    meta = json.loads((data / 'feature_map.json').read_text())
    features = {name: spec for group in meta['features'] for name, spec in group.items()}
    numerical = {name: spec for name, spec in features.items() if spec['type'] == 'numeric'}
    categorical = {name: spec for name, spec in features.items() if spec['type'] == 'categorical'}
    for name, spec in numerical.items():
        edges = np.asarray(spec.get('numeric_bin_edges', []), dtype=float)
        if len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError(f'Invalid training bin boundaries for {name}')
    report = dict(dataset=meta['dataset_id'], numerical_fields=len(numerical),
                  categorical_fields=len(categorical), synthetic=meta.get('synthetic', False),
                  feature_map_sha256=hashlib.sha256((data/'feature_map.json').read_bytes()).hexdigest(), splits={})
    for split in ('train', 'valid', 'test'):
        parquet = pq.ParquetFile(data / (split + '.parquet'))
        required = set(features) | set(meta['labels'])
        if not required.issubset(parquet.schema_arrow.names):
            raise ValueError(f'Missing columns: {required-set(parquet.schema_arrow.names)}')
        batch = next(parquet.iter_batches(batch_size=sample_rows, columns=list(required)))
        frame = batch.to_pandas()
        for name in numerical:
            if not np.isfinite(frame[name].to_numpy()).all():
                raise ValueError(f'Non-finite values: {split}/{name}')
        for name, spec in categorical.items():
            x = frame[name].to_numpy()
            if not ((x >= 0) & (x < spec['vocab_size']) & (x == np.floor(x))).all():
                raise ValueError(f'Invalid categorical IDs: {split}/{name}')
        for label in meta['labels']:
            if not frame[label].isin([0, 1]).all():
                raise ValueError(f'Non-binary labels: {split}/{label}')
        report['splits'][split] = dict(rows=parquet.metadata.num_rows, sampled_rows=len(frame),
                                       bytes=(data/(split+'.parquet')).stat().st_size)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--make-smoke', action='store_true')
    parser.add_argument('--train-rows', type=int, default=1024)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = make_smoke(args.dataset_dir, args.train_rows) if args.make_smoke else validate(args.dataset_dir)
    content = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == '__main__':
    main()
