"""Recompute paper statistics without importing PyTorch or requiring any dataset."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

METHODS = ('field', 'linear', 'linear_relu', 'quantile_bucket', 'log_squared_bucket',
           'autodis', 'ple_a', 'ple_b', 'ple_relu', 'periodic', 'plr', 'bspline',
           'narydis_lite', 'narydis_full', 'deer', 'dae', 'daes_gate', 'daes_tran', 'Fast')
BACKBONES = ('DNN', 'DeepFM', 'WideDeep', 'NFM', 'PNN', 'DCN', 'DCNv2', 'AutoInt', 'FiBiNET')
DATASETS = ('AutoML_A', 'AutoML_E', 'Criteo_x4_NumEmbRaw')
LABELS = ('Field', 'Linear', 'Linear-ReLU', 'Quantile bucket', 'Log-squared bucket',
          'AutoDis', 'PLE-A', 'PLE-B', 'PLE-ReLU', 'Periodic', 'PLR', 'B-spline',
          'NaryDis-Lite', 'NaryDis', 'DEER', 'DAE', 'DAES-Gate', 'DAES-Tran', 'ScalarLens')
METRICS = ('val_auc', 'val_logloss', 'test_auc', 'test_logloss')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trunc4(value):
    return str(Decimal(str(value)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))


def aggregate(rows, complete=True):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['dataset'], row['backbone'], row['method'], int(row['seed']))
        if key in seen:
            raise ValueError(f'Duplicate task: {key}')
        seen.add(key)
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f'Non-finite {metric}: {key}')
        if not 0 <= float(row['test_auc']) <= 1:
            raise ValueError(f'Invalid AUC: {key}')
        groups[key[:3]].append(row)
    if complete:
        expected = {(d, b, m, s) for d in DATASETS for b in BACKBONES for m in METHODS for s in (2026, 2027, 2028)}
        if seen != expected:
            raise ValueError(f'Unexpected task matrix: missing={len(expected-seen)}, extra={len(seen-expected)}')
    output = {}
    for key, group in groups.items():
        summary = dict(zip(('dataset', 'backbone', 'method'), key), n_seeds=len(group))
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[metric + '_mean'] = statistics.mean(values)
            summary[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.
        output[key] = summary
    for d in sorted({key[0] for key in output}):
        for b in sorted({key[1] for key in output if key[0] == d}):
            ordering = [*METHODS, 'narydis', 'scalarlens']
            methods = [m for m in ordering if (d, b, m) in output]
            # The original table uses this explicit order to break exact ties.
            for rank, m in enumerate(sorted(methods, key=lambda m: -output[d, b, m]['test_auc_mean']), 1):
                output[d, b, m]['rank_test_auc'] = rank
    return output


def audit_table(table_path, summaries):
    """Check all 513 AUC, SD and LL cells against the frozen LaTeX table."""
    dataset = None
    checked = []
    errors = []
    for line in Path(table_path).read_text().splitlines():
        for label, d in zip(('AutoML-A', 'AutoML-E', 'Criteo'), DATASETS):
            if r'\multicolumn{23}{l}{\textbf{' + label + '}}' in line:
                dataset = d
        if dataset is None or r'\aucstd' not in line:
            continue
        cells = line.split(' & ')
        title = re.sub(r'\\cite\{[^}]+\}', '', cells[0])
        title = title.replace(r'\textbf{', '').replace('}', '').replace('~', '').strip()
        if title not in LABELS:
            raise ValueError(f'Unknown paper method: {title}')
        method = METHODS[LABELS.index(title)]
        for i, backbone in enumerate(BACKBONES):
            nums = re.findall(r'(?:\d*\.\d+)', cells[1+2*i])
            ll = re.findall(r'\d+\.\d+', cells[2+2*i])[0]
            summary = summaries[dataset, backbone, method]
            expected = (trunc4(summary['test_auc_mean']), f"{summary['test_auc_std']:.4f}", trunc4(summary['test_logloss_mean']))
            actual = (nums[0], f'{float(nums[1]):.4f}', ll)
            record = dict(dataset=dataset, backbone=backbone, method=method,
                          auc=actual[0], auc_std=actual[1], logloss=actual[2], passed=actual == expected)
            checked.append(record)
            if actual != expected:
                errors.append(dict(record, expected=expected, actual=actual))
    if len(checked) != 513:
        raise ValueError(f'Expected 513 table cells, found {len(checked)}')
    return checked, errors


def reproduce(source, output, table=None, sample_seed=20261016):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = read_csv(source / 'per_seed_1539.csv')
    summaries = aggregate(per_seed)
    archived = read_csv(source / 'summary_513.csv')
    errors = []
    for row in archived:
        recomputed = summaries[row['dataset'], row['backbone'], row['method']]
        for metric in METRICS:
            for suffix in ('mean', 'std'):
                name = metric + '_' + suffix
                if abs(float(row[name]) - recomputed[name]) > 1e-12:
                    errors.append(f'{row["dataset"]}/{row["backbone"]}/{row["method"]}/{name}')
    table = Path(table) if table else source / 'table3_english.tex'
    checked, table_errors = audit_table(table, summaries)
    rng = random.Random(sample_seed)
    sample = []
    # Stratification covers all 27 dataset/backbone combinations. Include every
    # ScalarLens cell and one seeded random baseline cell per combination.
    for dataset in DATASETS:
        for backbone in BACKBONES:
            candidates = [row for row in checked if row['dataset'] == dataset and row['backbone'] == backbone]
            sample.append(next(row for row in candidates if row['method'] == 'Fast'))
            sample.append(rng.choice([row for row in candidates if row['method'] != 'Fast']))
    public_names = {'Fast': 'scalarlens', 'narydis_full': 'narydis'}
    canonical = lambda row: dict(row, method=public_names.get(row['method'], row['method']))
    write_csv(output / 'per_seed.csv', [canonical(row) for row in per_seed])
    ordered = [summaries[d, b, m] for d in DATASETS for b in BACKBONES for m in METHODS]
    write_csv(output / 'summary.csv', [canonical(row) for row in ordered])
    write_csv(output / 'sample_check.csv', [canonical(row) for row in sample])
    ranking = []
    for method in METHODS:
        ranks = [summaries[d, b, method]['rank_test_auc'] for d in DATASETS for b in BACKBONES]
        ranking.append(dict(method=public_names.get(method, method),
                            mean_rank=statistics.mean(ranks), wins=ranks.count(1), settings=len(ranks)))
    write_csv(output / 'ranking.csv', ranking)
    report = dict(status='PASS' if not errors and not table_errors else 'FAIL',
                  scope='Archived record reaggregation and paper-table audit; not a full training rerun.',
                  per_seed_records=len(per_seed), summary_groups=len(summaries),
                  aggregate_values_checked=len(archived)*len(METRICS)*2,
                  table_cells_checked=len(checked), table_values_checked=len(checked)*3,
                  stratified_sample_cells=len(sample), sample_seed=sample_seed,
                  aggregation_errors=errors, table_errors=table_errors,
                  source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ('per_seed_1539.csv', 'summary_513.csv', 'table3_english.tex')},
                  scalarlens=next(row for row in ranking if row['method'] == 'scalarlens'),
                  ranking_note='Ranks recomputed across the 19 displayed methods; archived rank columns are not used.')
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    text = ['# ScalarLens paper results', '',
            'Recomputed from archived per-seed CSV records; not newly trained results.', '',
            f"Verification: {report['status']}; 1,539 records, 513 groups, 54 stratified sample cells. All table AUC/SD/LL cells checked.", '',
            'Means are truncated to four decimals. Parentheses show sample SD over three seeds, rounded to four decimals. Ranks use untruncated AUC.', '']
    for dataset in DATASETS:
        text += [f'## {dataset}', '', '| Method | ' + ' | '.join(BACKBONES) + ' |',
                 '|---|' + '---|'*len(BACKBONES)]
        for method, label in zip(METHODS, LABELS):
            values = [summaries[dataset, b, method] for b in BACKBONES]
            text.append('| ' + label + ' | ' + ' | '.join(
                f"{trunc4(v['test_auc_mean'])} ({v['test_auc_std']:.4f}), LL {trunc4(v['test_logloss_mean'])}" for v in values) + ' |')
        text.append('')
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    if errors or table_errors:
        raise ValueError(f'Result audit failed; see {output / "verification.json"}')
    return report


def summarize_new_runs(runs, output, paper_source):
    """Summarize fresh task outputs separately and expose deltas, not a pass claim."""
    paths = sorted(Path(runs).rglob('result.json'))
    if not paths:
        raise ValueError(f'No result.json found under {runs}')
    rows = [json.loads(path.read_text()) for path in paths]
    groups = aggregate(rows, complete=False)
    archived = aggregate(read_csv(Path(paper_source)/'per_seed_1539.csv'))
    alias = {'scalarlens': 'Fast', 'narydis': 'narydis_full'}
    for key, group in groups.items():
        old = archived.get((key[0], key[1], alias.get(key[2], key[2])))
        group['paper_test_auc_mean'] = old['test_auc_mean'] if old else ''
        group['test_auc_delta_vs_paper'] = group['test_auc_mean']-old['test_auc_mean'] if old else ''
        group['paper_test_logloss_mean'] = old['test_logloss_mean'] if old else ''
        group['test_logloss_delta_vs_paper'] = group['test_logloss_mean']-old['test_logloss_mean'] if old else ''
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'summary.csv', [groups[key] for key in sorted(groups)])
    write_csv(output/'per_seed.csv', rows)
    lines = ['# New training results', '',
             'Only completed tasks are included. Groups with fewer than three seeds are incomplete. Deltas from the paper are observations, not automatic pass/fail decisions.', '',
             '| Dataset | Backbone | Method | Seeds | Test AUC mean | SD | Test LL mean | ΔAUC vs paper |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for key in sorted(groups):
        g = groups[key]
        delta = g['test_auc_delta_vs_paper']
        lines.append(f"| {g['dataset']} | {g['backbone']} | {g['method']} | {g['n_seeds']} | {g['test_auc_mean']:.6f} | {g['test_auc_std']:.6f} | {g['test_logloss_mean']:.6f} | {delta if delta == '' else format(delta, '+.6f')} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    return dict(tasks=len(rows), groups=len(groups), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # CLI outputs belong to the caller's workspace, never site-packages.
    root = Path.cwd()
    parser.add_argument('--source', type=Path, default=root / 'reproduction/paper')
    parser.add_argument('--output', type=Path, default=root / 'outputs/paper')
    parser.add_argument('--table', type=Path)
    parser.add_argument('--runs', type=Path, help='Summarize newly trained result.json files instead of auditing archived results')
    args = parser.parse_args()
    if args.runs:
        if args.output.resolve() == (root/'outputs/paper').resolve():
            parser.error('--runs requires an explicit separate --output directory')
        result = summarize_new_runs(args.runs, args.output, args.source)
    else:
        result = reproduce(args.source, args.output, args.table)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
