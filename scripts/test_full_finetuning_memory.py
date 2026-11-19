"""Measure one full fine-tuning epoch on MPS; never write a checkpoint.

Differences from the official script: one epoch, no validation data or metrics,
no results or plots, CPU checkpoint loading before moving the model to MPS,
and extra finite-value and memory checks. The training DataFrame is retained,
as in the official script. Sampled memory peaks can miss transient allocations.
"""

import argparse
import ast
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-rss-gib', type=float, default=None,
                        help='Stop before training if sampled process RSS exceeds this GiB limit')
    args = parser.parse_args()
    if args.max_rss_gib is not None and (not np.isfinite(args.max_rss_gib) or args.max_rss_gib <= 0):
        parser.error('--max-rss-gib must be a positive finite number')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; this test requires MPS and will not use CPU')

    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    device = torch.device('mps')
    process = psutil.Process()
    peak_rss = 0
    peak_mps = None
    completed = 0
    training_start = None
    print('Device: mps', flush=True)
    print('RSS and MPS allocations may overlap on unified memory; peaks are sampled.', flush=True)

    def report(stage, display=True):
        nonlocal peak_rss, peak_mps
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        mps = None
        if hasattr(torch.mps, 'current_allocated_memory'):
            mps = torch.mps.current_allocated_memory()
            peak_mps = mps if peak_mps is None else max(peak_mps, mps)
        if display:
            mps_text = f'{mps / 1024**3:.3f} GiB' if mps is not None else 'unavailable'
            print(f'{stage}: RSS {rss / 1024**3:.3f} GiB; MPS allocated {mps_text}', flush=True)
        return rss

    def check_limit(rss):
        if args.max_rss_gib is not None and rss > args.max_rss_gib * 1024**3:
            raise RuntimeError('RSS limit exceeded; stopped before training')

    def require_finite(tensors, description):
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise FloatingPointError(f'NaN/Inf detected in {description}')

    try:
        check_limit(report('Before loading'))
        csv_start = time.perf_counter()
        path = DATASET_DIR / 'tor-5tab-finetuning100way40shot.csv'
        df = pd.read_csv(path, header=None)
        csv_seconds = time.perf_counter() - csv_start
        check_limit(report('After CSV loading'))
        if df.shape != (4000, 15106):
            raise ValueError(f'Expected (4000, 15106), got {df.shape}')
        data = torch.tensor(df.iloc[:, 106:].values, dtype=torch.float32).unsqueeze(1)
        labels = torch.tensor(df.iloc[:, 0:100].values, dtype=torch.float32)
        check_limit(report('After tensor construction'))
        loader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=True)

        model_start = time.perf_counter()
        model = load_model_class()(num_classes=100)
        checkpoint = CHECKPOINT_DIR / 'FMWF_Pretrainmodel_weights200epoch.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device)
        torch.mps.synchronize()
        model_seconds = time.perf_counter() - model_start
        check_limit(report('After model and checkpoint loading'))
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        check_limit(report('Before training'))
        print(f'CSV loading time: {csv_seconds:.2f} s; model/checkpoint loading time: {model_seconds:.2f} s', flush=True)

        training_start = time.perf_counter()
        total_loss = 0.0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            require_finite((batch, target, output, loss), f'batch {completed + 1} input, labels, output or loss')
            report('After forward', display=False)
            loss.backward()
            require_finite((p.grad for p in model.parameters() if p.grad is not None),
                           f'batch {completed + 1} gradients')
            report('After backward', display=False)
            optimizer.step()
            require_finite(model.parameters(), f'batch {completed + 1} parameters')
            total_loss += loss.item()
            completed += 1
            report(f'Batch {completed}/{len(loader)}',
                   display=completed == 1 or completed % 25 == 0 or completed == len(loader))
        torch.mps.synchronize()
        training_seconds = time.perf_counter() - training_start
        print(f'Average training loss: {total_loss / len(loader):.6f}')
        print(f'Completed batches: {completed}/{len(loader)}; NaN/Inf detected: no')
        print(f'Training time: {training_seconds:.2f} s')
    except FloatingPointError:
        print('NaN/Inf detected: yes', flush=True)
        raise
    finally:
        report('Final observation')
        print(f'Completed batches: {completed}')
        print(f'Maximum observed RSS: {peak_rss / 1024**3:.3f} GiB')
        print('Maximum observed MPS allocated: ' +
              (f'{peak_mps / 1024**3:.3f} GiB' if peak_mps is not None else 'unavailable'))
        print(f'Total elapsed time (after imports): {time.perf_counter() - total_start:.2f} s')


if __name__ == '__main__':
    main()
