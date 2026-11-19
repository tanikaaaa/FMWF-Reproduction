"""Portable local-input paths; upstream code, raw data, and weights stay private."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = 'c970e0a38710f4bf260b5f63617cb59351af49cb'
UPSTREAM_DIR = Path(os.environ.get('FMWF_UPSTREAM_DIR', ROOT / 'upstream' / 'FMWF')).expanduser().resolve()
DATASET_DIR = Path(os.environ.get('FMWF_DATASET_DIR', ROOT / 'data' / 'tor5tab')).expanduser().resolve()
CHECKPOINT_DIR = Path(os.environ.get('FMWF_CHECKPOINT_DIR', ROOT / 'checkpoints')).expanduser().resolve()
METRICS_DIR = ROOT / 'results' / 'metrics'
