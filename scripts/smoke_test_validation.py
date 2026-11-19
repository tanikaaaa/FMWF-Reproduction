"""Check official top-five validation metrics on ten rows, without training."""

import ast
from pathlib import Path
from reproduction_paths import ROOT, UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, METRICS_DIR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


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


def calculate_metrics(all_targets, all_outputs):
    # Keep the official top-five selection and exact-set calculation unchanged.
    binarized_outputs = np.zeros_like(all_outputs)
    topk_indices = np.argsort(all_outputs, axis=1)[:, -5:]
    for i, indices in enumerate(topk_indices):
        binarized_outputs[i, indices] = 1
    val_accuracy = accuracy_score(all_targets, binarized_outputs)
    val_a_at_5 = np.mean(np.sum(np.take_along_axis(all_targets, topk_indices, axis=1) > 0, axis=1) / 5)
    return val_accuracy, val_a_at_5


def test_metrics():
    targets = np.zeros((1, 100), dtype=np.float32)
    targets[0, :5] = 1
    scores = np.zeros((1, 100), dtype=np.float32)
    # Predict four true sites (0-3), plus one false site (5), with no top-five ties.
    scores[0, [0, 1, 2, 3, 5]] = [10, 9, 8, 7, 6]
    exact, a_at_5 = calculate_metrics(targets, scores)
    if exact != 0 or not np.isclose(a_at_5, 0.8):
        raise RuntimeError(f'Metric-only test failed: exact={exact}, A@5={a_at_5}')
    print(f'Metric-only test: PASS; exact-set accuracy={exact:.1f}; A@5={a_at_5:.1f}', flush=True)


def main():
    test_metrics()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; run this test in the MPS-enabled VS Code environment')
    device = torch.device('mps')
    print(f'Device: {device}', flush=True)
    model = load_model_class()(num_classes=100)
    checkpoint = CHECKPOINT_DIR / 'FMWF_Pretrainmodel_weights200epoch.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    keys = model.load_state_dict(state, strict=True)
    print(f'Missing keys: {keys.missing_keys}; unexpected keys: {keys.unexpected_keys}', flush=True)
    del state
    model.to(device)

    path = DATASET_DIR / 'tor5tab-dataset-val0.csv'
    df = pd.read_csv(path, header=None, nrows=10)
    if df.shape != (10, 15106):
        raise ValueError(f'Expected (10, 15106), got {df.shape}')
    inputs = torch.tensor(df.iloc[:, 106:].values, dtype=torch.float32).unsqueeze(1).to(device)
    labels = torch.tensor(df.iloc[:, 0:100].values, dtype=torch.float32)
    targets = labels.numpy()
    if not np.all((targets == 0) | (targets == 1)) or not np.all(targets.sum(axis=1) == 5):
        raise ValueError('Expected binary multi-hot labels with exactly five positives per row')
    if not torch.isfinite(inputs).all().item():
        raise ValueError('NaN/Inf detected in traffic inputs')
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    print(f'Input shape: {tuple(inputs.shape)}; label shape: {tuple(labels.shape)}; '
          f'output shape: {tuple(outputs.shape)}', flush=True)
    if outputs.shape != (10, 100) or not torch.isfinite(outputs).all().item():
        raise RuntimeError('Invalid output shape or NaN/Inf in model output')
    exact, a_at_5 = calculate_metrics(targets, outputs.cpu().numpy())
    print(f'Exact-set accuracy: {exact:.6f}; A@5: {a_at_5:.6f}', flush=True)
    if not (0 <= exact <= 1 and 0 <= a_at_5 <= 1 and a_at_5 >= exact):
        raise RuntimeError('Metric bounds or A@5 >= exact-set accuracy check failed')
    print('Validation smoke test: PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Validation smoke test: FAIL', flush=True)
        raise
