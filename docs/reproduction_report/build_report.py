"""Build the editable Markdown report with matplotlib's existing PDF backend.

Run from any directory using the repository .venv Python. This utility reads saved
results and figures only. It does not import torch, train, or evaluate a model.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'reports' / 'FMWF_Reproduction_Report.pdf'
W, H = 595.276, 841.89
LEFT, RIGHT, TOP, BOTTOM = 55, 540.276, 72, 784
WIDTH = RIGHT - LEFT
INK, BLUE, MUTED = '#1E2933', '#175A7A', '#52616C'


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load_results():
    curves, evaluations, paths = {}, {}, {}
    for shot in (25, 40):
        curve = ROOT / 'results' / 'metrics' / f'100way-{shot}shot-validation_accuracy-300epoch.csv'
        curves[shot] = rows(curve)
        assert [int(r['epoch']) for r in curves[shot]] == list(range(40, 301))
        candidates = list((ROOT / 'results' / 'metrics').glob(
            f'FMWF_5tab_100way{shot}shot_finetuned300epoch_weights_evaluation_val0-val1-val2-val3-val4_*.csv'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one audited {shot}-shot evaluation, found {len(candidates)}')
        paths[shot] = candidates[0]
        evaluations[shot] = rows(paths[shot])
        sets = [r for r in evaluations[shot] if r['row_type'] == 'validation_set']
        assert [Path(r['validation_file']).stem for r in sets] == [f'tor5tab-dataset-val{i}' for i in range(5)]
        for key in ('Exact_set_accuracy', 'A@5'):
            mean = float(next(r[key] for r in evaluations[shot] if r['row_type'] == 'mean'))
            assert np.isclose(mean, np.mean([float(r[key]) for r in sets]), atol=1e-12)
            assert np.isclose(float(sets[0][key]), float(curves[shot][-1][key]), atol=1e-12)
    return curves, evaluations, paths


class Page:
    def __init__(self, number, total):
        self.number = number
        self.fig = plt.figure(figsize=(W / 72, H / 72), dpi=100, facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
        self.ax.axis('off')
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.y = TOP
        self.text('FMWF REPRODUCTION  /  TOR CLOSED-WORLD', LEFT, 34, 8, 'DejaVu Sans', 'bold', BLUE)
        self.ax.plot([LEFT, RIGHT], [49, 49], color='#CBD5DB', linewidth=0.7)
        self.ax.plot([LEFT, RIGHT], [802, 802], color='#CBD5DB', linewidth=0.7)
        self.text('B.Tech research project • evidence-based reproduction', LEFT, 812, 7.5, 'DejaVu Sans', color=MUTED)
        self.ax.text(RIGHT, 812, f'{number} / {total}', fontsize=8, color=MUTED, ha='right', va='top')

    def text(self, value, x, y, size=10.3, family='DejaVu Serif', weight='normal', color=INK):
        return self.ax.text(x, y, value, fontsize=size, family=family, weight=weight,
                            color=color, va='top', linespacing=1.4)

    def measure(self, value, size, family, weight):
        prop = FontProperties(family=family, size=size, weight=weight)
        return self.renderer.get_text_width_height_descent(value, prop, False)[0] * 72 / self.fig.dpi

    def wrap(self, value, width=WIDTH, size=10.3, family='DejaVu Serif', weight='normal'):
        lines, line = [], ''
        for word in value.split():
            proposed = (line + ' ' + word).strip()
            if self.measure(proposed, size, family, weight) <= width:
                line = proposed
                continue
            if line:
                lines.append(line)
            line = word
            # Long paths and filenames wrap without dropping any characters.
            while self.measure(line, size, family, weight) > width:
                cut = len(line) - 1
                while self.measure(line[:cut], size, family, weight) > width:
                    cut -= 1
                lines.append(line[:cut])
                line = line[cut:]
        if line:
            lines.append(line)
        return lines

    def paragraph(self, value, size=None, color=INK, family='DejaVu Serif', gap=9):
        size = size or (9.2 if self.number == 12 else 10.3)
        leading = size * 1.4
        lines = self.wrap(value, size=size, family=family)
        for line in lines:
            self.text(line, LEFT, self.y, size, family, color=color)
            self.y += leading
        self.y += gap

    def heading(self, value, level):
        size = (22 if level == 1 else 17) if self.number == 1 else (17 if level == 1 else 12)
        if level == 2:
            self.y += 5
        for line in self.wrap(value, size=size, family='DejaVu Sans', weight='bold'):
            self.text(line, LEFT, self.y, size, 'DejaVu Sans', 'bold', BLUE if level == 1 else INK)
            self.y += size * 1.25
        self.y += 12 if level == 1 else 7

    def table(self, data, weights=None):
        columns = len(data[0])
        weights = weights or [1] * columns
        widths = [WIDTH * w / sum(weights) for w in weights]
        for index, row in enumerate(data):
            size = 8.6 if columns >= 5 else 9
            weight = 'bold' if index == 0 else 'normal'
            wrapped = [self.wrap(str(cell), width=widths[i] - 12, size=size,
                                 family='DejaVu Sans', weight=weight) for i, cell in enumerate(row)]
            height = max(map(len, wrapped)) * 12 + 12
            self.ax.add_patch(Rectangle((LEFT, self.y), WIDTH, height,
                                       facecolor='#E5EFF4' if index == 0 else ('#F4F7F9' if index % 2 else 'white'),
                                       edgecolor='none'))
            x = LEFT
            for width, lines in zip(widths, wrapped):
                for j, line in enumerate(lines):
                    self.text(line, x + 6, self.y + 6 + j * 12, size, 'DejaVu Sans', weight)
                x += width
            self.y += height
            self.ax.plot([LEFT, RIGHT], [self.y, self.y], color='#D8E0E4', linewidth=0.5)
        self.y += 8

    def figure(self, name):
        path = ROOT / 'results' / 'figures' / name
        with Image.open(path) as im:
            iw, ih = im.size
        width = WIDTH
        height = width * ih / iw
        maxheight = 275 if self.number == 10 else (290 if self.number == 9 else 305)
        if height > maxheight:
            width *= maxheight / height
            height = maxheight
        x = LEFT + (WIDTH - width) / 2
        axes = self.fig.add_axes([x / W, 1 - (self.y + height) / H, width / W, height / H])
        axes.imshow(plt.imread(path))
        axes.axis('off')
        self.y += height + 12

    def callout(self, value):
        lines = self.wrap(value, width=WIDTH - 26, size=11, family='DejaVu Sans', weight='bold')
        height = len(lines) * 16 + 24
        self.ax.add_patch(Rectangle((LEFT, self.y), WIDTH, height, facecolor='#EDF4F7', edgecolor='none'))
        self.ax.add_patch(Rectangle((LEFT, self.y), 3, height, facecolor=BLUE, edgecolor='none'))
        for i, line in enumerate(lines):
            self.text(line, LEFT + 13, self.y + 10 + i * 16, 11, 'DejaVu Sans', 'bold', BLUE)
        self.y += height + 14

    def formula(self, value):
        self.ax.text(W / 2, self.y + 10, '$' + value + '$', fontsize=15, ha='center', va='top', color=INK)
        self.y += 62

    def pipeline(self):
        # Code-derived vector diagram; no generative imagery or paper diagram reuse.
        nodes = [('Synthetic CSV', '10,000 traces • 200 epochs'),
                 ('Shared CNN + MLP', '8,829,156 trainable parameters'),
                 ('Real few-shot adaptation', '2,500 / 4,000 traces • 300 epochs'),
                 ('Final checkpoint evaluation', '100 logits → top five → two metrics')]
        width, height = 330, 42
        x = (W - width) / 2
        for i, (title, detail) in enumerate(nodes):
            y = self.y + i * 59
            self.ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.5,rounding_size=5',
                                            facecolor='#EDF4F7', edgecolor='#8DADB9', linewidth=0.7))
            self.ax.text(W / 2, y + 7, title, ha='center', va='top', fontsize=10, weight='bold', color=BLUE)
            self.ax.text(W / 2, y + 23, detail, ha='center', va='top', fontsize=9, color=INK)
            if i < len(nodes) - 1:
                self.ax.add_patch(FancyArrowPatch((W / 2, y + height + 1), (W / 2, y + 58),
                                                  arrowstyle='-|>', mutation_scale=9, color=MUTED))
        self.y += 232


def dynamic(page, tag, evidence, curves, evaluations, paths):
    if tag == '@summary':
        table = [['Final epoch 300', 'Real samples', 'Mean A@5', 'Mean exact-set']]
        for shot in (25, 40):
            mean = next(r for r in evaluations[shot] if r['row_type'] == 'mean')
            table.append([f'{shot}-shot', f'{shot * 100:,}', f"{100 * float(mean['A@5']):.2f}%",
                          f"{100 * float(mean['Exact_set_accuracy']):.2f}%"])
        page.table(table, [1.2, 1, 1, 1.2])
    elif tag == '@software':
        versions = evidence['dependencies_snapshot']
        page.table([['Component', 'Inspected version', 'Provenance'],
                    ['Python / PyTorch', evidence['python_snapshot'] + ' / ' + versions['torch'], 'Local .venv; Torch also in E25/E40'],
                    ['NumPy / pandas', versions['numpy'] + ' / ' + versions['pandas'], 'Current repository .venv metadata'],
                    ['scikit-learn / matplotlib', versions['scikit-learn'] + ' / ' + versions['matplotlib'], 'Current repository .venv metadata'],
                    ['psutil', versions['psutil'], 'Current repository .venv; memory probes']], [1.25, 1, 1.7])
    elif tag in ('@best_a', '@best_exact'):
        key = 'A@5' if tag == '@best_a' else 'Exact_set_accuracy'
        table = [['Configuration', 'Final val0', 'Best observed val0', 'Epoch(s) of maximum']]
        for shot in (25, 40):
            best = max(float(r[key]) for r in curves[shot])
            epochs = [r['epoch'] for r in curves[shot] if float(r[key]) == best]
            table.append([f'{shot}-shot', f'{100 * float(curves[shot][-1][key]):.2f}%',
                          f'{100 * best:.2f}%', ', '.join(epochs)])
        page.table(table, [1, 1, 1.35, 1.4])
    elif tag == '@per_file':
        sets = {s: [r for r in evaluations[s] if r['row_type'] == 'validation_set'] for s in (25, 40)}
        table = [['File', 'n', '25-shot A@5', '25-shot exact', '40-shot A@5', '40-shot exact']]
        for i in range(5):
            table.append([f'val{i}', str(int(float(sets[25][i]['sample_count'])))] +
                         [f'{100 * float(sets[s][i][k]):.2f}' for s in (25, 40) for k in ('A@5', 'Exact_set_accuracy')])
        table.append(['Mean', '—'] + [f"{100 * float(next(r[k] for r in evaluations[s] if r['row_type'] == 'mean')):.2f}"
                                     for s in (25, 40) for k in ('A@5', 'Exact_set_accuracy')])
        page.table(table, [0.7, 0.5, 1.25, 1.25, 1.25, 1.25])
    elif tag == '@evaluation_sources':
        for shot in (25, 40):
            page.paragraph(f'[E{shot}] results/metrics/{paths[shot].name}. Rows 2–6 contain per-file results; row 7 is the mean; row 8 is population standard deviation. Checkpoint, file selection, device, seed, and Torch version are archived in every row.', size=9.2)
    elif tag == '@pipeline':
        page.pipeline()
    else:
        raise ValueError(tag)


def main():
    evidence = json.loads((HERE / 'evidence.json').read_text())
    if evidence['parameter_count'] != 8829156 or not all(evidence['model_class_ast_matches_commit'].values()):
        raise ValueError('Architecture evidence mismatch')
    curves, evaluations, paths = load_results()
    pages = (HERE / 'report.md').read_text().split('---PAGE---')
    if not 8 <= len(pages) <= 12:
        raise ValueError('Report page count is outside requested range')
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Serif',
                         'mathtext.fontset': 'dejavuserif', 'text.usetex': False})
    rendered = []
    for number, source in enumerate(pages, 1):
        page = Page(number, len(pages))
        blocks = re.split(r'\n\s*\n', source.strip())
        for block in blocks:
            block = block.strip()
            if block.startswith('# '):
                page.heading(block[2:], 1)
            elif block.startswith('## '):
                page.heading(block[3:], 2)
            elif block.startswith('|'):
                table = [[cell.strip() for cell in line.strip().strip('|').split('|')]
                         for line in block.splitlines()]
                # Give prose-heavy columns room to wrap without cramped typography.
                weights = [1, 3.2] if len(table[0]) == 2 else None
                if len(table[0]) == 3:
                    weights = [1.2, 2.5, 1.8] if number == 4 else [1.8, 1, 2.6]
                if number == 3 and len(table[0]) == 5:
                    weights = [1.65, 0.7, 0.8, 1.35, 1.2]
                page.table(table, weights)
            elif block.startswith('@figure '):
                page.figure(block.split(maxsplit=1)[1])
            elif block.startswith('@'):
                dynamic(page, block, evidence, curves, evaluations, paths)
            elif block.startswith('!callout '):
                page.callout(block[len('!callout '):])
            elif block.startswith('$$ '):
                page.formula(block[3:-3].strip())
            else:
                caption = block.startswith('Figure ')
                page.paragraph(block, size=9 if caption else None, color=MUTED if caption else INK,
                               family='DejaVu Sans' if caption else 'DejaVu Serif')
        print(f'Page {number}: body ends at {page.y:.1f} pt (limit {BOTTOM})')
        if page.y > BOTTOM:
            raise ValueError(f'Page {number} overflows by {page.y - BOTTOM:.1f} points; revise content/layout')
        rendered.append(page)
    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT, metadata={'Title': 'FMWF Reproduction Report',
                                'Author': 'B.Tech research project',
                                'Subject': 'Tor 5-tab 25-shot and 40-shot reproduction; evidence-based protocol audit',
                                'Keywords': 'FMWF, website fingerprinting, reproduction, MPS, A@5'}) as pdf:
        for page in rendered:
            pdf.savefig(page.fig, dpi=300)
            plt.close(page.fig)
    print(f'Saved {OUT} ({len(pages)} pages)')


if __name__ == '__main__':
    main()
