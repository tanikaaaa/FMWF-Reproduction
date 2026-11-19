"""Run one full 25-shot fine-tuning epoch, without validation or file outputs."""

import ast
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR
import random
import time

import numpy as np
import pandas as pd
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
    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(18)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = load_model_class()(num_classes=100)
    checkpoint = CHECKPOINT_DIR / 'FMWF_Pretrainmodel_weights200epoch.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)

    path = DATASET_DIR / 'tor-5tab-finetuning100way25shot.csv'
    df = pd.read_csv(path, header=None)
    if df.shape != (2500, 15106):
        raise ValueError(f'Expected dataset shape (2500, 15106), got {df.shape}')
    data = torch.tensor(df.iloc[:, 106:].values, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(df.iloc[:, 0:100].values, dtype=torch.float32)
    print(f'CSV shape: {df.shape}; input shape: {tuple(data.shape)}; '
          f'label shape: {tuple(labels.shape)}', flush=True)
    loader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=True, drop_last=False)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    if device.type == 'mps':
        torch.mps.synchronize()
    start = time.perf_counter()
    total_loss = 0.0
    completed_batches = 0
    seen_samples = 0
    final_batch_size = 0
    for batch, target in loader:
        batch, target = batch.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'NaN/Inf in training loss at batch {completed_batches + 1}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        completed_batches += 1
        final_batch_size = batch.shape[0]
        seen_samples += final_batch_size
    if device.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    if (seen_samples, completed_batches, final_batch_size) != (2500, 79, 4):
        raise RuntimeError(f'Incomplete epoch: samples={seen_samples}, '
                           f'batches={completed_batches}, final_batch={final_batch_size}')
    # Match the official script: average of batch losses, including the smaller last batch.
    print(f'Average training loss: {total_loss / len(loader):.6f}')
    print(f'Completed samples: {seen_samples}; batches: {completed_batches}; '
          f'final batch size: {final_batch_size}')
    print(f'Elapsed training time: {elapsed:.2f} seconds')
    print('NaN/Inf in training loss: no')
    print('One-epoch smoke test: PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('One-epoch smoke test: FAIL', flush=True)
        raise
