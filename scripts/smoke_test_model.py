"""Run a synthetic training step using the original pretraining model classes."""

import ast
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR

import torch
import torch.nn as nn


def load_model_class():
    source_path = UPSTREAM_DIR / 'FMWF-src' / 'PretrainingModel.py'
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in {'ConvBlock', 'CNNmodel'}]
    if [node.name for node in classes] != ['ConvBlock', 'CNNmodel']:
        raise RuntimeError('Expected exactly the original ConvBlock and CNNmodel classes')
    module = ast.Module(body=classes, type_ignores=[])
    namespace = {'nn': nn}
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['CNNmodel']


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    model = load_model_class()(num_classes=100).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.randn(2, 1, 15000, device=device)
    labels = torch.zeros(2, 100, device=device)
    labels[:, :5] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = model(inputs)
    print(f'Output shape: {tuple(outputs.shape)}', flush=True)
    if outputs.shape != (2, 100):
        raise RuntimeError(f'Unexpected output shape: {tuple(outputs.shape)}')
    loss = nn.MultiLabelSoftMarginLoss()(outputs, labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError('Loss is not finite')
    loss.backward()
    optimizer.step()
    if device.type == 'mps':
        torch.mps.synchronize()
    print(f'Loss: {loss.item():.6f}')
    print(f'Parameter count: {parameter_count:,}')
    print('Test: PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Test: FAIL', flush=True)
        raise
