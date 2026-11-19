"""Evaluate a final checkpoint (40-shot by default) on exactly five validation CSVs.

Default val0-val4 is provisional, not an identification of the paper's five sets.
Reports an unweighted arithmetic mean and population standard deviation (ddof=0)
across sets, even if their sample counts differ. No training or checkpoint writes.
Deterministic algorithms are required; unsupported operations fail explicitly.
"""

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = CHECKPOINT_DIR / 'FMWF_5tab_100way40shot_finetuned300epoch_weights.pth'


def load_model_class():
    path = UPSTREAM_DIR / 'FMWF-src' / 'tor-5tab-Accuracy.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in {'ConvBlock', 'CNNmodel'}]
    if [node.name for node in classes] != ['ConvBlock', 'CNNmodel']:
        raise RuntimeError('Expected exactly the original ConvBlock and CNNmodel classes')
    namespace = {'nn': nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CNNmodel']


def evaluate_file(model, device, path):
    df = pd.read_csv(path, header=None)
    if df.empty or df.shape[1] != 15106:
        raise ValueError(f'{path}: expected nonempty data with 15106 columns, got {df.shape}')
    inputs = torch.tensor(df.iloc[:, 106:].values, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(df.iloc[:, 0:100].values, dtype=torch.float32)
    if not torch.isfinite(inputs).all().item() or not torch.isfinite(labels).all().item():
        raise ValueError(f'{path}: nonfinite input or label values')
    targets = labels.numpy()
    if not np.all((targets == 0) | (targets == 1)) or not np.all(targets.sum(axis=1) == 5):
        raise ValueError(f'{path}: each label row must be binary with exactly five positives')
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=32, shuffle=False)
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            output = model(batch.to(device))
            if output.shape != (batch.shape[0], 100) or not torch.isfinite(output).all().item():
                raise RuntimeError(f'{path}: invalid output shape or nonfinite logits')
            all_outputs.append(output.cpu().numpy())
    all_outputs = np.vstack(all_outputs)
    # Match the official evaluator's top-five selection and exact-set accuracy.
    binarized_outputs = np.zeros_like(all_outputs)
    topk_indices = np.argsort(all_outputs, axis=1)[:, -5:]
    for i, indices in enumerate(topk_indices):
        binarized_outputs[i, indices] = 1
    exact = accuracy_score(targets, binarized_outputs)
    a_at_5 = np.mean(np.sum(np.take_along_axis(targets, topk_indices, axis=1) > 0, axis=1) / 5)
    if not (0 <= exact <= 1 and 0 <= a_at_5 <= 1 and a_at_5 >= exact):
        raise RuntimeError(f'{path}: invalid metric bounds or A@5 < exact-set accuracy')
    return len(df), float(exact), float(a_at_5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT,
                        help='Final checkpoint path; relative paths use the working directory (default: 40-shot)')
    parser.add_argument('--validation-dir', type=Path,
                        default=DATASET_DIR,
                        help='Base directory for relative validation filenames')
    parser.add_argument('--validation-files', nargs=5, metavar='CSV',
                        help='Exactly five distinct filenames relative to --validation-dir, or absolute paths')
    parser.add_argument('--output-dir', type=Path, default=METRICS_DIR,
                        help='Existing directory for a new uniquely named results CSV')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    provisional = args.validation_files is None
    names = args.validation_files or [f'tor5tab-dataset-val{i}.csv' for i in range(5)]
    files = [(Path(name) if Path(name).is_absolute() else args.validation_dir / name).resolve()
             for name in names]
    if len(set(files)) != 5:
        parser.error('Select five distinct validation files')
    for path in files:
        if not path.is_file():
            parser.error(f'Validation file does not exist: {path}')
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')
    if not args.output_dir.is_dir():
        parser.error('--output-dir must be an existing directory')

    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    torch.use_deterministic_algorithms(True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}; deterministic algorithms required', flush=True)
    print('Default val0-val4 is PROVISIONAL; the paper\'s exact five sets are unverified.'
          if provisional else 'Using five explicitly selected validation files.', flush=True)
    model = load_model_class()(num_classes=100)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    keys = model.load_state_dict(state, strict=True)
    print(f'Missing keys: {keys.missing_keys}; unexpected keys: {keys.unexpected_keys}', flush=True)
    del state
    model.to(device)
    model.eval()

    rows = []
    for path in files:
        count, exact, a_at_5 = evaluate_file(model, device, path)
        print(f'{path.name}: samples={count}; exact-set accuracy={exact:.6f}; A@5={a_at_5:.6f}', flush=True)
        rows.append({'row_type': 'validation_set', 'validation_file': path.name,
                     'sample_count': count, 'Exact_set_accuracy': exact, 'A@5': a_at_5})
    exact_values = np.array([row['Exact_set_accuracy'] for row in rows])
    a_values = np.array([row['A@5'] for row in rows])
    for kind, exact, a_at_5 in (
            ('mean', exact_values.mean(), a_values.mean()),
            ('population_std_ddof0', exact_values.std(ddof=0), a_values.std(ddof=0))):
        print(f'Across five sets ({kind}): exact-set accuracy={exact:.6f}; A@5={a_at_5:.6f}')
        rows.append({'row_type': kind, 'validation_file': '', 'sample_count': None,
                     'Exact_set_accuracy': exact, 'A@5': a_at_5})
    for row in rows:
        row.update(checkpoint=checkpoint.name, selected_validation_files=';'.join(path.name for path in files),
                   provisional_default_selection=provisional, device=str(device), seed=18,
                   torch_version=str(torch.__version__))

    tags = [re.sub(r'[^A-Za-z0-9_-]', '_', path.stem.removeprefix('tor5tab-dataset-'))[:16]
            for path in files]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir / f'{checkpoint.stem}_evaluation_{"-".join(tags)}_{stamp}.csv'
    # Exclusive creation refuses any overwrite, including an unlikely timestamp collision.
    with output.open('x', newline='') as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
    print(f'Saved new results CSV: {output}')


if __name__ == '__main__':
    main()
