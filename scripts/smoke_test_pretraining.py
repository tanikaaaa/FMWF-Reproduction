"""Test two pretraining epochs on the first 64 synthetic dataset rows."""

import ast
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR
import random

import numpy as np
import pandas as pd
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
    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    path = DATASET_DIR / 'synthesis-5tabdataset.csv'
    df = pd.read_csv(path, header=None, nrows=64)
    if df.shape != (64, 15100):
        raise ValueError(f'Expected 64 rows and 15100 columns, got {df.shape}')
    data = torch.tensor(df.iloc[:, 100:].values, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(df.iloc[:, :100].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=True)
    model = load_model_class()(num_classes=100).to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    peak = None
    nonfinite = False

    def sample_memory():
        nonlocal peak
        if device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
            value = torch.mps.current_allocated_memory()
            peak = value if peak is None else max(peak, value)

    def check_finite(tensors):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

    try:
        sample_memory()
        model.train()
        for epoch in range(2):
            total_loss = 0.0
            for batch, target in loader:
                batch, target = batch.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(batch)
                sample_memory()
                if output.shape != (32, 100):
                    raise RuntimeError(f'Unexpected output shape: {tuple(output.shape)}')
                loss = criterion(output, target)
                nonfinite |= not check_finite((batch, target, output, loss))
                loss.backward()
                sample_memory()
                nonfinite |= not check_finite(p.grad for p in model.parameters() if p.grad is not None)
                optimizer.step()
                sample_memory()
                nonfinite |= not check_finite(model.parameters())
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/2 average loss: {total_loss / len(loader):.6f}', flush=True)
        if nonfinite:
            raise RuntimeError('NaN or Inf detected during training')
        print('Forward, backward, and optimizer steps: PASS')
    finally:
        print(f'NaN/Inf detected: {"yes" if nonfinite else "no"}')
        if peak is None:
            print('Peak MPS allocated memory: unavailable')
        else:
            print(f'Peak sampled MPS allocated memory: {peak / (1024 ** 2):.2f} MiB '
                  '(sampled between steps; transient peak may be higher)')


if __name__ == '__main__':
    main()
