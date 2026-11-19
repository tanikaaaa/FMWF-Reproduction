"""Measure a single full-data pretraining epoch; no checkpoints are written.

The CSV and column slices match PretrainingModel.py. After tensor construction,
the DataFrame and NumPy views are released; the official script retains them and
also retains epoch outputs for metrics. This test omits those metric buffers.
"""

import argparse
import ast
import gc
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR
import random

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_model_class():
    path = UPSTREAM_DIR / 'FMWF-src' / 'PretrainingModel.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in {'ConvBlock', 'CNNmodel'}]
    if [node.name for node in classes] != ['ConvBlock', 'CNNmodel']:
        raise RuntimeError('Expected exactly the original ConvBlock and CNNmodel classes')
    namespace = {'nn': nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CNNmodel']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-rss-gib', type=float, default=None,
                        help='Stop before training if process RSS exceeds this GiB limit; '
                             'checked after CSV and tensor loading (default: no limit).')
    args = parser.parse_args()
    if args.max_rss_gib is not None and args.max_rss_gib <= 0:
        parser.error('--max-rss-gib must be positive')

    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    process = psutil.Process()
    peak_rss = 0
    peak_mps = None
    print(f'Device: {device}', flush=True)
    print('Memory values are sampled; process RSS and MPS allocations may overlap on unified memory.', flush=True)

    def report(stage):
        nonlocal peak_rss, peak_mps
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        mps = None
        if device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
            mps = torch.mps.current_allocated_memory()
            peak_mps = mps if peak_mps is None else max(peak_mps, mps)
        mps_text = f'{mps / 1024**3:.3f} GiB' if mps is not None else 'unavailable'
        print(f'{stage}: RSS {rss / 1024**3:.3f} GiB; MPS allocated {mps_text}', flush=True)
        return rss

    def check_limit(rss):
        if args.max_rss_gib is not None and rss > args.max_rss_gib * 1024**3:
            raise RuntimeError('RSS limit exceeded; stopped before training')

    def finite(tensors):
        return all(torch.isfinite(t).all().item() for t in tensors)

    try:
        path = DATASET_DIR / 'synthesis-5tabdataset.csv'
        df = pd.read_csv(path, header=None)
        check_limit(report('After CSV loading'))
        if df.shape != (10000, 15100):
            raise ValueError(f'Expected (10000, 15100), got {df.shape}')
        features = df.iloc[:, 100:].values
        labels = df.iloc[:, 0:100].values
        data = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor(labels, dtype=torch.float32)
        check_limit(report('After tensor construction'))
        del df, features, labels
        gc.collect()
        check_limit(report('After releasing CSV data'))

        loader = DataLoader(TensorDataset(data, targets), batch_size=32, shuffle=True)
        model = load_model_class()(num_classes=100).to(device)
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        check_limit(report('After model construction, before training'))
        model.train()
        total_loss = 0.0
        for batch_index, (batch, target) in enumerate(loader, 1):
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            if not finite((batch, target, output, loss)):
                raise RuntimeError(f'NaN/Inf detected in input, target, output or loss at batch {batch_index}')
            loss.backward()
            if not finite(p.grad for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f'NaN/Inf detected in gradients at batch {batch_index}')
            optimizer.step()
            if not finite(model.parameters()):
                raise RuntimeError(f'NaN/Inf detected in parameters at batch {batch_index}')
            total_loss += loss.item()
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                report(f'Batch {batch_index}/{len(loader)}')
        print(f'Average training loss: {total_loss / len(loader):.6f}')
        print('NaN/Inf detected: no; one epoch completed')
    finally:
        report('Final observation')
        print(f'Maximum observed RSS: {peak_rss / 1024**3:.3f} GiB')
        print('Maximum observed MPS allocated: ' +
              (f'{peak_mps / 1024**3:.3f} GiB' if peak_mps is not None else 'unavailable'))


if __name__ == '__main__':
    main()
