# Reproducibility guide

## Scope and prerequisites

This bundle contains saved aggregate results and our support utilities. It does
not redistribute the authors' architecture/training files, model weights, raw
traffic, or the published paper. Obtain the upstream repository directly, check
out commit `c970e0a38710f4bf260b5f63617cb59351af49cb`, and configure the three
`FMWF_*_DIR` variables described in the root README.

All architecture loaders parse only `ConvBlock` and `CNNmodel` via Python AST,
then instantiate those definitions without executing upstream top-level training.
The 25-shot one-epoch utility loads these identical model classes from the
official `tor-5tab-Accuracy.py`, since the locally derived 25-shot full-training
script is intentionally excluded. The original model ASTs were verified equal.

Checkpoint loading in these utilities uses CPU `map_location`, restricted
`weights_only=True`, and strict state-dictionary loading where applicable.
Inputs remain float32, labels occupy `0:100`, real traffic occupies `106:`, and
synthetic traffic occupies `100:`. No personal home-directory path is required.

## Commands and effects

Run from the reproduction root with your chosen Python environment. These commands
are documentation only; staging preparation did not execute model tests or evaluation.

| Utility in `scripts/` | Runtime effect |
|---|---|
| `smoke_test_model.py` | Synthetic batch of 2, one forward/backward/Adam step; MPS or CPU |
| `smoke_test_pretraining.py` | First 64 synthetic rows, two training epochs; MPS or CPU |
| `smoke_test_finetuning.py` | First 64 real 40-shot rows, two epochs from pretrained weights |
| `smoke_test_validation.py` | Metric-only deterministic example and 10 val0 rows; requires MPS; no training |
| `smoke_test_25shot_full_epoch.py` | All 2,500 real 25-shot rows, one epoch, 79 batches including final 4 |
| `test_full_dataset_memory.py` | All 10,000 synthetic rows, one epoch; optional pre-training RSS limit |
| `test_full_finetuning_memory.py` | All 4,000 real 40-shot rows, one epoch; timing/RSS; requires MPS |
| `evaluate_5tab_validation_sets.py` | Five CSV files, no training, deterministic eval/no_grad; MPS or CPU |
| `plot_final_reproduction_results.py` | Reads saved result CSVs and writes figures; no model/data loading |

For a memory probe, `--max-rss-gib 10` refuses training if a sampled pre-training
RSS reading exceeds the threshold. It cannot prevent an intra-stage spike and
does not enforce a training-time memory ceiling. RSS and MPS allocations can
overlap on unified memory. The pretraining probe releases some buffers retained
by the full official script; its measurements are not an upper bound for that run.

Evaluate an explicit 25-shot checkpoint on the provisional default files:

```bash
python scripts/evaluate_5tab_validation_sets.py \
  --checkpoint "$FMWF_CHECKPOINT_DIR/FMWF_5tab_100way25shot_finetuned300epoch_weights.pth"
```

The default checkpoint is the final 40-shot file. To choose five files explicitly:

```bash
python scripts/evaluate_5tab_validation_sets.py \
  --validation-dir "$FMWF_DATASET_DIR" \
  --validation-files tor5tab-dataset-val0.csv tor5tab-dataset-val1.csv \
    tor5tab-dataset-val2.csv tor5tab-dataset-val3.csv tor5tab-dataset-val4.csv \
  --output-dir results/metrics
```

Every output is timestamped and exclusively created. Newly generated evaluation
metadata stores filenames rather than private absolute paths. External files with
identical basenames are not uniquely identified by that metadata; preserve local
hashes if using alternative inputs. The included audit manifest documents the
original inputs, and selected provisional files must not be relabeled as the
authors' confirmed evaluation files.

Plotting refuses to overwrite existing figures. `--figure-3-only` selects only
the Figure 3 v2 outputs, which are already included; a refusal in that case is
intentional. `--evaluation-25` and `--evaluation-40` select saved measurement CSVs.
No alternative run is selected automatically when multiple matching files exist.

## Full training, obtained separately

No full-training author source or copied 25-shot derivative is included. In your
separate upstream checkout, the audited configuration is:

- Pretraining: complete 10,000-row synthetic CSV, 200 epochs.
- Fine-tuning: separate initialization from original pretrained weights for each
  support file; 2,500 rows for 25 shots or 4,000 for 40 shots; 300 epochs.
- Seed 18; float32 `(B,1,15000)`; batch size 32; shuffled training; Adam 0.001;
  `MultiLabelSoftMarginLoss`; all model parameters trainable.
- Validation: val0 loaded once, batch size 32, eval/no_grad after every epoch 40–300.
- Reproduction additions: local dataset paths; MPS-compatible device selection and
  checkpoint mapping; CUDA seed guard; A@5 beside unchanged exact-set accuracy;
  selected-device printing; separate final state dictionaries and result CSVs.

The unmodified upstream script has CUDA/CPU selection and root-level file paths.
It computes exact-set accuracy and does not save a fine-tuned checkpoint. Therefore
running it unchanged is not a turnkey reproduction of our MPS artifact workflow.
Review and implement those documented local adaptations in your private checkout
under appropriate permissions. This bundle intentionally contains no contextual
source patch while upstream redistribution permission is unresolved.

## Regenerate the report or inspect evidence

`python docs/reproduction_report/build_report.py` rebuilds the publication report
from sanitized CSVs, evidence, and existing figures; no model weights are required.
It replaces only `reports/FMWF_Reproduction_Report.pdf`.

`collect_evidence.py` audits locally supplied datasets/checkpoints and an upstream
checkout, without inference. It writes **ignored**
`local-artifacts/report_evidence.json`, leaving the archived public evidence intact.
The optional `FMWF_PAPER_PATH` points to a separately obtained local paper.

Exact result reproduction remains conditional on access to the same data and
weights. Cross-device/PyTorch-version equality is not guaranteed by deterministic
algorithm selection alone. Unsupported deterministic MPS operations raise errors.
