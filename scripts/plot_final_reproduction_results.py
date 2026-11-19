"""Plot saved FMWF results only; never load models or evaluate traffic.

Creates four PNG/PDF figure pairs under FMWF-results when explicitly executed.
Curves are unsmoothed, final checkpoints are epoch 300, and val0-val4 remains a
provisional selection. The paper headline is a descriptive reference, not a
protocol-matched comparison. Population standard deviations are descriptive
across overlapping validation files, not uncertainty estimates for independent runs.
"""

import argparse
import csv
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


SOURCE = METRICS_DIR
RESULTS = ROOT / 'results' / 'figures'
COLORS = {25: '#0072B2', 40: '#D55E00'}
MARKERS = {25: 'o', 40: 's'}
VAL_NAMES = [f'tor5tab-dataset-val{i}.csv' for i in range(5)]
PAPER_PERCENT = 91.0  # Approximate headline: local published paper, p. 1069.


def read_rows(path, required):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {required - set(reader.fieldnames or [])}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty results')
    return rows


def metrics(rows, path):
    values = np.array([[float(r['Exact_set_accuracy']), float(r['A@5'])] for r in rows])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f'{path}: metrics must be finite fractions in [0, 1]')
    if (values[:, 1] + 1e-12 < values[:, 0]).any():
        raise ValueError(f'{path}: A@5 is below exact-set accuracy')
    return values


def load_curve(shot):
    path = SOURCE / f'100way-{shot}shot-validation_accuracy-300epoch.csv'
    rows = read_rows(path, {'epoch', 'Exact_set_accuracy', 'A@5'})
    epochs = np.array([float(row['epoch']) for row in rows])
    if not np.array_equal(epochs, np.arange(40, 301)):
        raise ValueError(f'{path}: expected exactly one saved row for every epoch 40-300, in order')
    return epochs, metrics(rows, path), path


def evaluation_path(shot, explicit):
    if explicit is not None:
        return explicit.resolve()
    pattern = (f'FMWF_5tab_100way{shot}shot_finetuned300epoch_weights_'
               'evaluation_val0-val1-val2-val3-val4_*.csv')
    matches = sorted(SOURCE.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Expected one {shot}-shot evaluation CSV, found {len(matches)}; '
                         f'specify --evaluation-{shot} explicitly (no automatic run selection)')
    return matches[0]


def load_evaluation(shot, path):
    required = {'row_type', 'validation_file', 'sample_count', 'Exact_set_accuracy',
                'A@5', 'checkpoint', 'selected_validation_files', 'provisional_default_selection'}
    rows = read_rows(path, required)
    checkpoint = f'FMWF_5tab_100way{shot}shot_finetuned300epoch_weights.pth'
    if any(Path(row['checkpoint']).name != checkpoint for row in rows):
        raise ValueError(f'{path}: checkpoint provenance does not match {shot}-shot epoch 300')
    if any([Path(p).name for p in row['selected_validation_files'].split(';')] != VAL_NAMES
           for row in rows):
        raise ValueError(f'{path}: selected sets must be val0-val4 in that order')
    sets = [r for r in rows if r['row_type'] == 'validation_set']
    if len(sets) != 5 or len({Path(r['validation_file']).name for r in sets}) != 5:
        raise ValueError(f'{path}: expected five distinct validation-set rows')
    lookup = {Path(r['validation_file']).name: r for r in sets}
    if set(lookup) != set(VAL_NAMES):
        raise ValueError(f'{path}: validation files differ from val0-val4')
    ordered = [lookup[name] for name in VAL_NAMES]
    counts = np.array([float(r['sample_count']) for r in ordered])
    if not np.array_equal(counts, [100, 100, 100, 200, 100]):
        raise ValueError(f'{path}: sample counts differ from the audited datasets')
    values = metrics(ordered, path)
    means = [r for r in rows if r['row_type'] == 'mean']
    stds = [r for r in rows if r['row_type'] == 'population_std_ddof0']
    if len(means) != 1 or len(stds) != 1 or len(rows) != 7:
        raise ValueError(f'{path}: expected five sets, one mean, and one population-standard-deviation row')
    saved_mean = metrics(means, path)[0]
    saved_std = np.array([float(stds[0][k]) for k in ('Exact_set_accuracy', 'A@5')])
    if not np.allclose(saved_mean, values.mean(axis=0), rtol=0, atol=1e-12):
        raise ValueError(f'{path}: saved mean disagrees with unweighted set mean')
    if not np.allclose(saved_std, values.std(axis=0, ddof=0), rtol=0, atol=1e-12):
        raise ValueError(f'{path}: saved standard deviation disagrees with set values')
    return values, counts.astype(int), saved_mean


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#D9DEE3', linewidth=0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def save_figure(fig, name, caption, sources, left=0.12, top=0.86):
    fig.subplots_adjust(left=left, right=0.97, bottom=0.27, top=top)
    fig.text(0.12, 0.055, caption, ha='left', va='bottom', fontsize=9, color='#404850', linespacing=1.5)
    for extension in ('png', 'pdf'):
        destination = RESULTS / f'{name}.{extension}'
        # Exclusive creation protects any plots from an earlier run.
        with destination.open('xb') as handle:
            metadata = {'Title': name, 'Description': caption + '\nSources: ' + '; '.join(Path(str(path)).name for path in sources)}
            if extension == 'pdf':
                metadata = {'Title': name, 'Subject': metadata['Description']}
            fig.savefig(handle, format=extension, dpi=300, facecolor='white', metadata=metadata)
        print(f'Saved {destination}')
    plt.close(fig)


def plot_curve(curves, metric_index, number, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for shot in (25, 40):
        epochs, values, _ = curves[shot]
        percent = 100 * values[:, metric_index]
        ax.plot(epochs, percent, color=COLORS[shot], linewidth=1.25,
                label=f'{shot}-shot | epoch 300: {percent[-1]:.2f}%')
        ax.scatter([300], [percent[-1]], color=COLORS[shot], marker=MARKERS[shot],
                   s=45, edgecolor='white', linewidth=0.7, zorder=4, clip_on=False)
    ax.set(xlim=(40, 300), ylim=(0, 100), xlabel='Fine-tuning epoch', ylabel=ylabel, title=title)
    ax.set_xticks([40, 80, 120, 160, 200, 240, 280, 300])
    style_axes(ax)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    save_figure(fig, f'figure_{number}_val0_{"a_at_5" if metric_index else "exact_set_accuracy"}_by_epoch',
                'Single-set val0 validation; 100-way Tor 5-tab. All saved epochs shown without smoothing.\n'
                'Markers identify epoch 300; no best-epoch selection. Exact-set accuracy and A@5 are distinct.',
                [curves[s][2] for s in (25, 40)])


def plot_sets(evaluations, paths):
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    x = np.arange(5)
    for shot, offset in ((25, -0.19), (40, 0.19)):
        values, counts, _ = evaluations[shot]
        bars = ax.bar(x + offset, 100 * values[:, 1], width=0.36, color=COLORS[shot], label=f'{shot}-shot')
        ax.bar_label(bars, labels=[f'{v * 100:.1f}' for v in values[:, 1]], padding=3, fontsize=9)
    ax.set_xticks(x, [f'val{i}\nn = {n}' for i, n in enumerate(counts)])
    ax.set(ylim=(0, 104), ylabel='A@5 (%)', xlabel='Validation file',
           title='Final epoch-300 checkpoint A@5 by validation file')
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.545, 0.98),
               ncol=2, frameon=False)
    save_figure(fig, 'figure_3_final_a_at_5_by_validation_file_v2',
                'Provisional val0-val4 selection; the paper\'s exact five validation sets are unverified.\n'
                'All 100 samples in val4 are duplicated within val3 (n = 200); files are not disjoint.', paths,
                top=0.80)


def plot_paper_reference(evaluations, paths):
    reproduced = 100 * evaluations[25][2][1]
    gap = PAPER_PERCENT - reproduced
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.scatter([PAPER_PERCENT], [1], color='#69737D', marker='D', s=90, zorder=3)
    ax.scatter([reproduced], [0], color=COLORS[25], marker='o', s=90, zorder=3)
    ax.set_yticks([0, 1], ['Reproduction\n25-shot, final epoch 300', 'Paper headline\n25-shot, approximately'])
    ax.text(reproduced + 0.18, 0, f'{reproduced:.2f}%', va='center', fontsize=11)
    ax.text(PAPER_PERCENT + 0.18, 1, f'~{PAPER_PERCENT:.1f}%', va='center', fontsize=11, color='#69737D')
    ax.annotate('', xy=(reproduced, 0.5), xytext=(PAPER_PERCENT, 0.5),
                arrowprops={'arrowstyle': '<->', 'color': '#404850', 'lw': 1.2})
    ax.text((reproduced + PAPER_PERCENT) / 2, 0.6,
            f'~{gap:.2f} percentage-point numerical gap', ha='center', fontsize=10)
    ax.set(xlim=(84, 96), ylim=(-0.5, 1.5), xlabel='A@5 (%)',
           title='Paper reference and reproduced 25-shot result')
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', color='#D9DEE3', linewidth=0.65)
    ax.set_axisbelow(True)
    save_figure(fig, 'figure_4_paper_reference_vs_reproduction_25shot',
                'Descriptive reference only: evaluation protocols are NOT verified to match.\n'
                'Reproduction: unweighted mean of provisional, overlapping val0-val4; final checkpoint.\n'
                'Paper: approximately 91.0%; exact sets and checkpoint-selection policy are undocumented.',
                [paths[0], 'Beyond Single Tabs (WWW 2025), p. 1069; Appendix A.2, p. 1077'], left=0.31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-25', type=Path, help='Explicit saved 25-shot five-set results CSV')
    parser.add_argument('--evaluation-40', type=Path, help='Explicit saved 40-shot five-set results CSV')
    parser.add_argument('--figure-3-only', action='store_true',
                        help='Create only Figure 3 v2, leaving all other figure files untouched')
    args = parser.parse_args()
    curves = {shot: load_curve(shot) for shot in (25, 40)}
    paths = [evaluation_path(25, args.evaluation_25), evaluation_path(40, args.evaluation_40)]
    evaluations = {shot: load_evaluation(shot, path) for shot, path in zip((25, 40), paths)}
    for shot in (25, 40):
        if not np.allclose(curves[shot][1][-1], evaluations[shot][0][0], rtol=0, atol=1e-12):
            raise ValueError(f'{shot}-shot epoch-300 val0 curve differs from saved final evaluation')
    if not np.allclose([evaluations[s][2][1] for s in (25, 40)], [0.8872, 0.9212], rtol=0, atol=1e-12):
        raise ValueError('Saved means differ from the audited reproduction results; review inputs')
    if args.figure_3_only:
        filenames = ['figure_3_final_a_at_5_by_validation_file_v2']
    else:
        filenames = ['figure_1_val0_a_at_5_by_epoch', 'figure_2_val0_exact_set_accuracy_by_epoch',
                     'figure_3_final_a_at_5_by_validation_file_v2', 'figure_4_paper_reference_vs_reproduction_25shot']
    collisions = [RESULTS / f'{name}.{ext}' for name in filenames for ext in ('png', 'pdf')
                  if (RESULTS / f'{name}.{ext}').exists()]
    if collisions:
        raise FileExistsError('Refusing to overwrite existing outputs for the selected mode:\n' +
                              '\n'.join(str(path) for path in collisions))
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                         'axes.titlepad': 14, 'axes.labelsize': 11, 'axes.linewidth': 0.8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})
    if args.figure_3_only:
        plot_sets(evaluations, paths)
        return
    plot_curve(curves, 1, 1, 'Fine-tuning validation A@5 on val0', 'A@5 (%)')
    plot_curve(curves, 0, 2, 'Fine-tuning exact-set accuracy on val0', 'Exact-set accuracy (%)')
    plot_sets(evaluations, paths)
    plot_paper_reference(evaluations, paths)


if __name__ == '__main__':
    main()
