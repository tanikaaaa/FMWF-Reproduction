"""Fine-tune the official CNN for two epochs on 64 real traces, without saving."""

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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = load_model_class()(num_classes=100)
    checkpoint = CHECKPOINT_DIR / 'FMWF_Pretrainmodel_weights200epoch.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected_keys = set(model.state_dict())
    checkpoint_keys = set(state)
    print(f'Missing checkpoint keys: {sorted(expected_keys - checkpoint_keys)}', flush=True)
    print(f'Unexpected checkpoint keys: {sorted(checkpoint_keys - expected_keys)}', flush=True)
    model.load_state_dict(state, strict=True)
    model.to(device)

    path = DATASET_DIR / 'tor-5tab-finetuning100way40shot.csv'
    df = pd.read_csv(path, header=None, nrows=64)
    if df.shape != (64, 15106):
        raise ValueError(f'Expected (64, 15106), got {df.shape}')
    inputs = torch.tensor(df.iloc[:, 106:].values, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(df.iloc[:, 0:100].values, dtype=torch.float32)
    print(f'Input shape: {tuple(inputs.shape)}; label shape: {tuple(labels.shape)}', flush=True)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=32, shuffle=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    def require_finite(tensors, description):
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise FloatingPointError(f'NaN/Inf detected in {description}')

    try:
        for epoch in range(2):
            total_loss = 0.0
            for batch, target in loader:
                batch, target = batch.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(batch)
                print(f'Batch input shape: {tuple(batch.shape)}; output shape: {tuple(output.shape)}', flush=True)
                if output.shape != (32, 100):
                    raise RuntimeError(f'Unexpected output shape: {tuple(output.shape)}')
                loss = criterion(output, target)
                require_finite((batch, target, output, loss), 'input, labels, output or loss')
                loss.backward()
                require_finite((p.grad for p in model.parameters() if p.grad is not None), 'gradients')
                optimizer.step()
                require_finite(model.parameters(), 'model parameters')
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/2 average loss: {total_loss / len(loader):.6f}', flush=True)
    except FloatingPointError:
        print('NaN/Inf detected: yes', flush=True)
        raise
    print('NaN/Inf detected: no')
    print('Fine-tuning smoke test: PASS')


if __name__ == '__main__':
    main()
