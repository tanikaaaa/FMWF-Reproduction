"""Read-only source/data audit for the report; never runs model inference/training."""
import array
import ast
import csv
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduction_paths import UPSTREAM_DIR, DATASET_DIR, CHECKPOINT_DIR, UPSTREAM_COMMIT
DATA = DATASET_DIR
PAPER = Path(os.environ['FMWF_PAPER_PATH']).expanduser() if 'FMWF_PAPER_PATH' in os.environ else None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_dataset(path, offset, hash_inputs=False):
    rows = 0
    widths, cardinalities, values, repetitions, combinations = set(), set(), set(), set(), set()
    hashes, full_hashes = set(), set()
    invalid_labels = 0
    combination_counts = Counter()
    with path.open(newline='') as handle:
        for row in csv.reader(handle):
            rows += 1
            widths.add(len(row))
            labels = tuple(float(x) for x in row[:100])
            combinations.add(labels)
            combination_counts[labels] += 1
            invalid_labels += int(any(v not in (0, 1) for v in labels))
            cardinalities.add(sum(labels))
            values.update(row[offset:])
            if offset == 106:
                repetitions.add(float(row[105]))
            if hash_inputs:
                hashes.add(hashlib.sha256(array.array('f', map(float, row[offset:])).tobytes()).digest())
                full_hashes.add(hashlib.sha256(array.array('f', map(float, row)).tobytes()).digest())
    result = dict(rows=rows, column_counts=sorted(widths), positive_labels_per_row=sorted(cardinalities),
                  invalid_binary_label_rows=invalid_labels, traffic_values=sorted(values),
                  unique_label_combinations=len(combinations),
                  samples_per_label_combination=sorted(set(combination_counts.values())),
                  metadata106_values=sorted(repetitions))
    print(path.name, rows, 'rows', flush=True)
    return result, hashes, full_hashes


def main():
    evidence = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM_DIR, text=True).strip(),
                'python_snapshot': platform.python_version(), 'dependencies_snapshot': {}, 'datasets': {}}
    if evidence['commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('Upstream checkout must be at the documented commit')
    for name in ('torch', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'psutil', 'pillow'):
        evidence['dependencies_snapshot'][name] = importlib.metadata.version(name)
    synthetic = DATA / 'synthesis-5tabdataset.csv'
    evidence['datasets'][synthetic.name] = audit_dataset(synthetic, 100)[0]
    training = {}
    for shot in (25, 40):
        path = DATA / f'tor-5tab-finetuning100way{shot}shot.csv'
        info, hashes, _ = audit_dataset(path, 106, True)
        evidence['datasets'][path.name] = info
        training[shot] = hashes
    validation = {}
    for i in range(13):
        path = DATA / f'tor5tab-dataset-val{i}.csv'
        info, hashes, full = audit_dataset(path, 106, True)
        info['exact_input_overlap_with_training'] = {str(s): len(hashes & training[s]) for s in (25, 40)}
        evidence['datasets'][path.name] = info
        validation[i] = (hashes, full)
    evidence['validation_cross_file_overlap'] = [
        {'files': [j, i], 'exact_float32_input_matches': len(validation[i][0] & validation[j][0]),
         'full_numeric_row_matches': len(validation[i][1] & validation[j][1])}
        for i in range(13) for j in range(i) if validation[i][0] & validation[j][0]]

    import torch
    import torch.nn as nn
    original = subprocess.check_output(['git', 'show', 'HEAD:FMWF-src/tor-5tab-Accuracy.py'], cwd=UPSTREAM_DIR, text=True)
    classes = [n for n in ast.parse(original).body if isinstance(n, ast.ClassDef)]
    namespace = {'nn': nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<committed-model-classes>', 'exec'), namespace)
    model = namespace['CNNmodel'](100)
    evidence['parameter_count'] = sum(p.numel() for p in model.parameters())
    evidence['model_class_ast_matches_commit'] = {}
    for name in ('PretrainingModel.py', 'tor-5tab-Accuracy.py'):
        current = [n for n in ast.parse((UPSTREAM_DIR / 'FMWF-src' / name).read_text()).body if isinstance(n, ast.ClassDef)]
        evidence['model_class_ast_matches_commit'][name] = [ast.dump(n) for n in current] == [ast.dump(n) for n in classes]
    evidence['strict_checkpoint_checks'] = {}
    for name in ('FMWF_Pretrainmodel_weights200epoch.pth',
                 'FMWF_5tab_100way25shot_finetuned300epoch_weights.pth',
                 'FMWF_5tab_100way40shot_finetuned300epoch_weights.pth'):
        state = torch.load(CHECKPOINT_DIR / name, map_location='cpu', weights_only=True)
        result = model.load_state_dict(state, strict=True)
        evidence['strict_checkpoint_checks'][name] = dict(missing=result.missing_keys, unexpected=result.unexpected_keys)
        del state

    inputs = list((ROOT / 'scripts').glob('*.py'))
    inputs += list((ROOT / 'results' / 'metrics').glob('*.csv'))
    inputs += list((ROOT / 'results' / 'figures').glob('*'))
    if PAPER is not None:
        inputs.append(PAPER)
    def public_key(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return 'private-input/' + path.name
    evidence['source_sha256'] = {public_key(p): digest(p) for p in inputs if p.is_file()}
    output = ROOT / 'local-artifacts' / 'report_evidence.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('Saved local-artifacts/report_evidence.json; no training or inference executed', flush=True)


if __name__ == '__main__':
    main()
