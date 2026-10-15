#!/usr/bin/env python3
"""Collect baseline and adaptive-split results into CSV, JSON, Markdown and a figure.

Usage:
    python adaptive_report.py --data ~/lwctrans-work/data --out ~/lwctrans-work/data/adaptive_split
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

DS = 'Dataset402_AISD'
METRICS = ['Dice', 'IoU', 'HD95', 'ASSD', 'Precision', 'Sensitivity', 'Specificity']
DISTANCE = {'HD95', 'ASSD'}
MODELS = [
    ('baseline_50_50', 'nnUNetTrainer', 'Fixed 50:50 split (LW-CTrans default, our run)'),
    ('ratio', 'nnUNetTrainerSplitRatio', 'Learnable static split ratio'),
    ('dynratio', 'nnUNetTrainerSplitDynRatio', 'Input-dependent split ratio'),
    ('gate', 'nnUNetTrainerSplitGate', 'Fixed 50:50 split + input-dependent branch gate'),
]
PAPER = {'model': 'paper_50_50', 'description': 'LW-CTrans paper, Table 7, best ratio (50:50 both stages)',
         'protocol': 'paper (5-fold CV)', 'params_M': 2.08, 'test_Dice_mean': 62.35, 'test_Dice_std': 19.51}
BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked', '.tau')


def nan_stats(values, scale):
    a = np.array([np.nan if v is None else float(v) for v in values], dtype=float) * scale
    if a.size == 0 or np.all(np.isnan(a)):
        return None, None
    return float(np.nanmean(a)), float(np.nanstd(a))


def case_id(path):
    return Path(path).name.replace('.nii.gz', '')


def per_case(summary_path):
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text())
    out = {}
    for c in data['metric_per_case']:
        m = c['metrics'].get('1', c['metrics'].get(1))
        out[case_id(c['reference_file'])] = {k: m.get(k) for k in METRICS}
    return out


def ckpt_info(fold):
    info = {}
    path = next((fold / n for n in ('checkpoint_final.pth', 'checkpoint_best.pth') if (fold / n).is_file()), None)
    if path is None:
        return info
    try:
        import torch
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        info['ckpt_error'] = str(e)
        return info
    sd = ck['network_weights']
    n = sum(v.numel() for k, v in sd.items() if not k.endswith(BUFFER_SUFFIXES))
    info['params'] = int(n)
    info['params_M'] = round(n / 1e6, 4)
    log = ck.get('logging', {})
    st, en = log.get('epoch_start_timestamps', []), log.get('epoch_end_timestamps', [])
    if st and en:
        k = min(len(st), len(en))
        times = [en[i] - st[i] for i in range(k)]
        info['epochs_trained'] = k
        info['mean_epoch_s'] = round(float(np.mean(times)), 2)
        info['train_hours'] = round(float(np.sum(times)) / 3600, 3)
    ema = log.get('ema_fg_dice', [])
    if ema:
        info['best_ema_pseudo_dice'] = round(float(np.max(ema)), 4)
        info['best_epoch'] = int(np.argmax(ema))
    return info


def paired(base, other):
    keys = [k for k in base if k in other]
    pairs = [(base[k]['Dice'], other[k]['Dice']) for k in keys]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)]
    if not pairs:
        return {}
    a = np.array([p[0] for p in pairs]) * 100
    b = np.array([p[1] for p in pairs]) * 100
    d = b - a
    res = {'n_pairs': len(pairs), 'mean_diff': round(float(d.mean()), 3), 'median_diff': round(float(np.median(d)), 3),
           'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum())}
    try:
        from scipy.stats import wilcoxon
        res['wilcoxon_p'] = float(wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0
    except Exception as e:
        res['wilcoxon_p'] = None
        res['wilcoxon_note'] = str(e)
    return res


def split_info(fold):
    out = {}
    fj = fold / 'split_final.json'
    if fj.is_file():
        out['final'] = json.loads(fj.read_text())
    sc = fold / 'split_stats.csv'
    if sc.is_file():
        rows = list(csv.DictReader(sc.open()))
        if rows:
            out['last_val_epoch'] = {k: float(v) for k, v in rows[-1].items()}
            out['history'] = [{k: float(v) for k, v in r.items()} for r in rows]
    return out


def fmt(m, s, nd=2):
    return '' if m is None else f'{m:.{nd}f} ± {s:.{nd}f}'


def plot_curves(models, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False
    series = [(r['model'], r['split'].get('history', [])) for r in models if r.get('split', {}).get('history')]
    if not series:
        return False
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 3.6), squeeze=False)
    for ax, (name, hist) in zip(axes[0], series):
        ep = [h['epoch'] for h in hist]
        for key in sorted(hist[0]):
            if key.endswith('_rho_mean') or key.endswith('_alpha_mean'):
                ax.plot(ep, [h[key] for h in hist], label=key.replace('_mean', ''))
                std_key = key.replace('_mean', '_std') if key.endswith('_rho_mean') else key.replace('_mean', '_sample_std')
                if std_key in hist[0]:
                    m = np.array([h[key] for h in hist]); s = np.array([h[std_key] for h in hist])
                    ax.fill_between(ep, m - s, m + s, alpha=0.2)
        ax.axhline(0.5, color='grey', lw=0.8, ls='--')
        ax.set_title(name)
        ax.set_xlabel('epoch')
        ax.set_ylabel('CNN share (rho) or gate alpha')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path.home() / 'lwctrans-work' / 'data'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--results', default=os.environ.get('nnUNet_results', ''))
    ap.add_argument('--outpath', default='LW-CTrans')
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    out = Path(args.out).expanduser() if args.out else data / 'adaptive_split'
    results = Path(args.results).expanduser() if args.results else data / 'nnUNet_results'
    rep = out / 'report'
    rep.mkdir(parents=True, exist_ok=True)

    rows, cases = [], {}
    for name, trainer, desc in MODELS:
        fold = results / DS / f'{trainer}__nnUNetPlans__3d_fullres' / args.outpath / 'fold_0'
        test_json = data / 'test_eval_summary.json' if name == 'baseline_50_50' else out / name / 'test_eval_summary.json'
        val = per_case(fold / 'validation' / 'summary.json')
        test = per_case(test_json)
        if val is None and test is None:
            print(f'skip {name}: no results yet ({fold})')
            continue
        r = {'model': name, 'trainer': trainer, 'description': desc, 'protocol': 'ours (fold 0, 150 epochs, checkpoint_best for test)'}
        r.update(ckpt_info(fold))
        for split, pc in (('val', val), ('test', test)):
            if pc is None:
                continue
            r[f'{split}_n'] = len(pc)
            for m in METRICS:
                mean, std = nan_stats([v[m] for v in pc.values()], 1 if m in DISTANCE else 100)
                r[f'{split}_{m}_mean'] = None if mean is None else round(mean, 3)
                r[f'{split}_{m}_std'] = None if std is None else round(std, 3)
        r['split'] = split_info(fold)
        cases[name] = {'val': val, 'test': test}
        rows.append(r)

    if not rows:
        print('nothing to report yet')
        return

    tests = []
    base = cases.get('baseline_50_50')
    if base:
        for r in rows:
            if r['model'] == 'baseline_50_50':
                continue
            for split in ('val', 'test'):
                if base[split] and cases[r['model']][split]:
                    t = paired(base[split], cases[r['model']][split])
                    if t:
                        tests.append({'model': r['model'], 'vs': 'baseline_50_50', 'split': split, **t})

    flat_keys = ['model', 'description', 'protocol', 'params', 'params_M', 'epochs_trained', 'mean_epoch_s', 'train_hours',
                 'best_epoch', 'best_ema_pseudo_dice', 'val_n'] + [f'val_{m}_{s}' for m in METRICS for s in ('mean', 'std')] + \
                ['test_n'] + [f'test_{m}_{s}' for m in METRICS for s in ('mean', 'std')]
    with (rep / 'summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({k: PAPER.get(k, '') for k in flat_keys})

    with (rep / 'per_case.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'split', 'case_id'] + METRICS)
        for name, sp in cases.items():
            for split in ('val', 'test'):
                for cid, m in sorted((sp[split] or {}).items()):
                    w.writerow([name, split, cid] + [m[k] for k in METRICS])

    if tests:
        keys = ['model', 'vs', 'split', 'n_pairs', 'mean_diff', 'median_diff', 'wins', 'losses', 'ties', 'wilcoxon_p']
        with (rep / 'paired_tests.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(tests)

    with (rep / 'learned_split.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'source', 'key', 'value'])
        for r in rows:
            sp = r.get('split', {})
            for stage, vals in sp.get('final', {}).get('blocks', {}).items():
                for k, v in vals.items():
                    w.writerow([r['model'], 'final_weights', f'{stage}_{k}', v])
            for k, v in sp.get('last_val_epoch', {}).items():
                w.writerow([r['model'], 'last_val_epoch', k, v])

    has_fig = plot_curves(rows, rep / 'split_curves.png')

    json_rows = [{k: v for k, v in r.items() if k != 'split'} | {'split_final': r.get('split', {}).get('final'),
                 'split_last_val_epoch': r.get('split', {}).get('last_val_epoch')} for r in rows]
    (rep / 'summary.json').write_text(json.dumps({'models': json_rows, 'paper_reference': PAPER, 'paired_tests': tests},
                                                 indent=2, default=str))

    md = ['# LW-CTrans adaptive channel split on AISD', '',
          'Our runs: fold 0 of nnU-Net 5-fold split (272 train / 68 val), 150 epochs, one run per model. '
          'Test set: 52 held-out AISD cases, predicted with checkpoint_best. The paper row uses 5-fold CV and is not directly comparable.', '',
          '| Model | Params (M) | Val Dice (%) | Test Dice (%) | Test IoU (%) | Test HD95 (mm) | Test Precision (%) | Test Sensitivity (%) |',
          '|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['model']} | {r.get('params_M', '')} | {fmt(r.get('val_Dice_mean'), r.get('val_Dice_std'))} | "
                  f"{fmt(r.get('test_Dice_mean'), r.get('test_Dice_std'))} | {fmt(r.get('test_IoU_mean'), r.get('test_IoU_std'))} | "
                  f"{fmt(r.get('test_HD95_mean'), r.get('test_HD95_std'))} | {fmt(r.get('test_Precision_mean'), r.get('test_Precision_std'))} | "
                  f"{fmt(r.get('test_Sensitivity_mean'), r.get('test_Sensitivity_std'))} |")
    md.append(f"| {PAPER['model']} | {PAPER['params_M']} | | {PAPER['test_Dice_mean']:.2f} ± {PAPER['test_Dice_std']:.2f} | | | | |")
    if tests:
        md += ['', '## Paired comparison against our 50:50 baseline (per-case Dice, percentage points)', '',
               '| Model | Split | Pairs | Mean diff | Median diff | Wins / Losses | Wilcoxon p |', '|---|---|---|---|---|---|---|']
        for t in tests:
            p = '' if t.get('wilcoxon_p') is None else f"{t['wilcoxon_p']:.4f}"
            md.append(f"| {t['model']} | {t['split']} | {t['n_pairs']} | {t['mean_diff']:+.2f} | {t['median_diff']:+.2f} | "
                      f"{t['wins']} / {t['losses']} | {p} |")
    md += ['', '## Learned split', '']
    for r in rows:
        fin = r.get('split', {}).get('final', {}).get('blocks')
        last = r.get('split', {}).get('last_val_epoch')
        if fin or last:
            md.append(f"- **{r['model']}**: final={json.dumps(fin)}; last validation epoch={json.dumps(last)}")
    if has_fig:
        md += ['', '![split curves](split_curves.png)']
    (rep / 'summary.md').write_text('\n'.join(md) + '\n')

    print('\n'.join(md))
    print(f'\nreport written to {rep}')


if __name__ == '__main__':
    main()
