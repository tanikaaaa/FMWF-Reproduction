# FMWF Reproduction Report

## Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks

Original authors: Wenwen Meng, Chuan Ma, Ming Ding, Chunpeng Ge, Yuwen Qian, and Tao Xiang. Proceedings of the ACM Web Conference 2025 (WWW '25), pp. 1068–1077. DOI: 10.1145/3696410.3714811. [P1]

B.Tech research project | Technical reproduction dossier | 19 November 2026

Repository: https://github.com/WW-Meng/FMWF

Audited base commit: c970e0a38710f4bf260b5f63617cb59351af49cb

## Reproduction summary

This report documents completed Tor closed-world experiments with five concurrent tabs, a 100-dimensional website-label space, and the released FMWF CNN. Synthetic pretraining used 10,000 traces for 200 epochs. Two separate models were then initialized from that pretrained state and fine-tuned for 300 epochs using the official 25-shot and 40-shot CSVs, with 2,500 and 4,000 real traces respectively. The available logs identify MPS as the execution device. [S1–S3, L1–L3]

@summary

The reported final results are unweighted means of five file-level scores on provisional val0–val4. All checkpoints are from epoch 300; the means are not selected from the best validation epochs. The paper's approximately 91.0% Tor five-tab 25-shot headline is 2.28 percentage points above the reproduced 88.72%. That numerical difference is not a strictly protocol-matched performance difference. [P1, E25, E40]

!callout Metric-matched and broadly configuration-matched reproduction, but not fully evaluation-protocol-matched.

The authors' five validation files and checkpoint-selection policy are unidentified. The selected files overlap: all 100 val4 rows also occur in the 200-row val3 file. This report preserves that limitation rather than describing the files as independent tests. No additional training or inference was performed to prepare the report.

---PAGE---
# 1. Research problem and implemented method

Website fingerprinting infers visited websites from observable encrypted-traffic patterns. Encryption conceals content but does not necessarily conceal packet direction, sequence structure, or the timing and volume of communication. In multi-tab browsing, several websites contribute to a mixed trace. A single-tab classifier's assumption that one input belongs to one website is therefore insufficient: the model must recognize multiple website labels from one observation. [P1, §§1–2]

FMWF combines synthetic pretraining and few-shot transfer to reduce the need for large, freshly collected multi-tab training sets. The paper describes concatenating single-tab traces into synthetic zero-time-interval traffic sequences and assigning multi-hot labels. Real overlapping traces are then used to adapt the pretrained feature extractor and classifier. This reproduction uses the provided synthetic CSV; the repository does not provide a complete generator that would establish its exact sampling and truncation provenance. [P1, §3.2; R1, lines 25–29]

@pipeline

Figure A. Reproduction pipeline derived from the executable code. The CNN and MLP are the same in pretraining and fine-tuning; all trainable parameters are updated during fine-tuning. Final evaluation loads saved state dictionaries without executing the training modules. [S1–S4]

## What was actually reproduced

The input is a fixed-length float32 tensor with shape (batch, 1, 15000). The model emits 100 raw logits, while each real trace has five positive entries in its 100-dimensional multi-hot target. The CNN does not segment the trace into separate websites. A multilabel objective trains the network jointly, and validation ranks website scores to select five predictions. [S1, lines 81–104; S2, lines 24–73 and 89–121]

The paper also covers other tab counts, browsers, unknown tab counts, different label-space sizes, open-world traffic, and revised augmentation. Those experiments are outside this report. No comparison measurements for TMWF, ARES, or Forge were generated here; later cross-method comparison requires a shared dataset and evaluation protocol.

---PAGE---
# 2. Datasets and execution environment

All dataset CSVs are read with header=None, so their first rows are observations. Synthetic labels occupy columns 0:100 and traffic occupies 100:. Real labels occupy 0:100; six metadata columns are skipped and traffic occupies 106:. These are zero-based Python slices. The report's read-only schema audit verifies the following dimensions and binary five-positive targets. [S1–S3; D1]

| Dataset | Samples | CSV columns | Input tensor | Label tensor |
| Synthetic pretraining | 10,000 | 15,100 | (10000, 1, 15000) | (10000, 100) |
| Real 25-shot support | 2,500 | 15,106 | (2500, 1, 15000) | (2500, 100) |
| Real 40-shot support | 4,000 | 15,106 | (4000, 1, 15000) | (4000, 100) |
| val0, val1, val2, val4 | 100 each | 15,106 | (100, 1, 15000) | (100, 100) |
| val3 | 200 | 15,106 | (200, 1, 15000) | (200, 100) |

The stored traffic values are direction/padding codes −1, 0, and +1, converted to float32 without changing length. “100-way” follows the released dataset naming and paper terminology. The output encodes 100 websites; the real support files contain 100 distinct five-website combinations repeated 25 or 40 times, rather than 100 mutually exclusive single-site classes. [D1]

## Platform and software provenance

The hardware is an Apple Silicon MacBook Air with 16 GB unified memory, as reported by the operator. The precise chip generation and macOS build were not archived. Pretraining and both full fine-tuning logs explicitly record MPS. Final evaluation CSV provenance records device=mps, seed=18, and PyTorch 2.3.0. [L1–L3, E25, E40]

@software

Dependency versions in the table are the inspected repository .venv snapshot, not a contemporaneous lockfile for every run. Python 3.12.16 also agrees with the operator's experiment context; only PyTorch 2.3.0 is independently stored in the evaluation CSVs. Exact historical versions of the other packages cannot be proven. The official README instead lists Ubuntu 22.04, RTX 3080 Ti, CUDA 12.1, Python 3.12, and PyTorch 2.3.0. [R1, lines 6–13]

---PAGE---
# 3. Exact model architecture and training

The released ConvBlock and CNNmodel definitions were extracted without executing top-level training. Their abstract syntax trees match the corresponding classes in the current pretraining, 40-shot, and 25-shot scripts. The verified trainable parameter count is 8,829,156 (approximately 8.83 million). [S0–S3; D1]

| Stage | Structure | Output for length 15,000 |
| Block 1 | Conv 1→32→32; ELU | (B, 32, 3749) |
| Block 2 | Conv 32→64→64; ReLU | (B, 64, 936) |
| Block 3 | Conv 64→128→128; ReLU | (B, 128, 233) |
| Block 4 | Conv 128→256→256; ReLU | (B, 256, 57) |
| Flatten | 256 × 57 activations | (B, 14592) |
| Dense 1 | Linear 14592→512, no bias; BN; ReLU; dropout 0.7 | (B, 512) |
| Dense 2 | Linear 512→512, no bias; BN; ReLU; dropout 0.3 | (B, 512) |
| Output | Linear 512→100, with bias; no output activation | (B, 100) |

Every block contains two bias-free Conv1d layers (kernel 8, stride 1, padding 4), each followed by BatchNorm1d and the block's activation. MaxPool1d uses kernel 8, stride 4, padding 0; block dropout is 0.1. Activations are in-place. The fixed flattened dimension constrains acceptable sequence lengths. [S0, lines 23–71; S2, lines 24–73]

| Setting | Synthetic pretraining | Real fine-tuning |
| Epochs | 200 | 300 per configuration |
| Batch size | 32, shuffle=True | 32, shuffle=True |
| Optimizer / learning rate | Adam / 0.001 | Adam / 0.001 |
| Objective | MultiLabelSoftMarginLoss() | MultiLabelSoftMarginLoss() |
| Seed / dtype | 18 / float32 | 18 / float32 |
| Updates | Entire CNN and classifier | Entire CNN and classifier |

Fine-tuning creates a new Adam optimizer over model.parameters(); no feature blocks are frozen and no separate classifier-only stage occurs. Checkpoint weights initialize the model, but optimizer state is not resumed. There is no early stopping or scheduler. The 25-shot loader produces 79 batches, including a final batch of four; the 40-shot loader produces 125 full batches. Epoch loss follows the released mean of batch losses, so the last small batch has equal batch weight. [S2, lines 118–148; S3, lines 121–150]

---PAGE---
# 4. Reproduction process and artifact lineage

The reproduction preserved the official architecture and training settings while making the data paths and device handling usable on macOS. Pretraining selects CUDA, then MPS, then CPU; CUDA-specific seeding is guarded. Dataset paths resolve to the supplied Tor directory. Output paths resolve to the scripts' directory. Fine-tuning checkpoint loading uses map_location. The separate 25-shot script selects MPS with CPU fallback. [S1–S3]

| Step | Evidence and interpretation |
| Read-only data inspection | Dimensions, five-positive targets, signed traffic codes, and exact-input overlap were audited without changing CSVs. |
| Synthetic pretraining | L1 contains 200 epoch records, input shapes, and mps. Final training loss is 0.0005; this is a rounded training metric, not held-out performance. |
| Smoke tests | Standalone scripts extract model classes via AST, avoiding top-level CSV loading and training. Earlier Codex synthetic smoke testing used CPU; operator MPS tests are separate evidence. |
| Full-data probes | Operator reported complete one-epoch MPS tests, no NaN/Inf, pretraining RSS 3.912 GiB, and 40-shot RSS 1.877 GiB with 16.48 s for 125 batches. Raw probe console logs were not found in the repository. |
| Full fine-tuning | L2 and L3 each contain 300 training records and 261 val0 validations, with startup device mps. |
| Checkpoint evaluation | E25 and E40 record final checkpoint paths, MPS, seed 18, five selected files, per-file metrics, mean, and population standard deviation. |

The memory figures are sampled process RSS, not peak total unified-memory usage. MPS allocations and RSS can overlap. The pretraining probe releases the DataFrame and omits training-metric buffers retained by the official script; its observation is not an upper bound on the official run. Probe timing includes diagnostic checks and is not a full-run benchmark. [S5; operator-reported observations]

## Checkpoint handling

Pretraining saves FMWF_Pretrainmodel_weights200epoch.pth and the original full-model filename. Both fine-tuning configurations initialize from the pretrained state dictionary, never from each other. The reproduction adds separate final 25-shot and 40-shot state-dictionary filenames after epoch 300. The report audit strictly loaded all three state dictionaries on CPU with no missing or unexpected keys; it ran no forward pass. [S1, lines 152–154; S2, lines 85–87 and 196–201; S3, lines 85–87 and 199–204; D1]

---PAGE---
# 5. Evaluation definitions and protocol

Let y be the binary 100-dimensional target vector, and let rK(z) be the indices of the K largest website scores. The paper defines the fraction of these predictions that correspond to visited websites. The per-file result is the mean across samples. [P1, Appendix A.2, equation (1), printed p. 1077]

$$ A@K(x)=\frac{1}{K}\sum_{i\in r_K(z(x))}y_i $$

For this five-tab experiment, K=5. Our implementation uses np.argsort to obtain five indices, counts positive target entries at those indices, divides by five, and averages over all rows. Logits suffice for ranking because applying softmax preserves their order. The code does not change the output activation or the training loss to calculate this metric. [S2, lines 171–180; S4, lines 61–70]

$$ \mathrm{ExactSet}=\frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\{\hat{\mathbf{y}}_n=\mathbf{y}_n\} $$

The released sklearn accuracy_score on a multilabel indicator matrix gives exact-set accuracy: every selected website must match for a sample to count as correct. With five true labels and four correct predictions, A@5 is 0.8 but exact-set accuracy is 0. The original exact-set calculation was retained, and A@5 was added beside it using identical top-five predictions. [S0, lines 165–172; S2, lines 171–180]

## Validation schedule and aggregation

Each fine-tuning run loads val0 once, calls model.eval() with torch.no_grad() after epoch 40 and every later epoch, then returns to model.train(). Exactly 261 epoch-level validation results are recorded for val0, corresponding to evaluations after epochs 40 through 300. The subsequent standalone evaluator strictly loads the saved epoch-300 state, requires deterministic algorithms, uses eval/no_grad, and evaluates each selected file with shuffle=False. [S2, lines 105–123 and 150–194; S4, lines 40–70 and 100–126]

$$ \overline{A@5}=\frac{1}{5}\sum_{j=0}^{4} A@5_j $$

The five-file mean is unweighted: val3's 200 samples do not give it twice the file weight. The reported population standard deviation uses ddof=0 across the five file scores. It is descriptive spread, not a standard error or confidence interval. Files overlap, and each configuration has one documented training trajectory; this report makes no multi-run significance claim.

---PAGE---
# 6. Validation A@5 over fine-tuning

@figure figure_1_val0_a_at_5_by_epoch.png

Figure 1. Saved, unsmoothed A@5 on single-set val0 for every epoch from 40 through 300. The marked final values are 89.00% for 25-shot and 91.20% for 40-shot. These single-set values differ from the five-file means. The embedded figure is the existing final artifact, not a newly generated curve. [C25, C40, F1]

@best_a

Best observed A@5 is a descriptive property of the logged val0 curve. The 25-shot maximum occurs at epochs 187 and 189; the 40-shot maximum occurs at epoch 233. The final-checkpoint results elsewhere in this report do not substitute these maxima, and no best-epoch checkpoint is available from the current saving procedure. [C25, C40]

The difference between the final and best val0 observations illustrates sensitivity to checkpoint choice; it does not establish that the paper selected a best epoch. The released script logs repeated validation but does not document a rule for converting that curve into the headline paper result. Val0 was also observed repeatedly during training, so this report calls it validation rather than a previously untouched test set.

---PAGE---
# 7. Original-code exact-set accuracy

@figure figure_2_val0_exact_set_accuracy_by_epoch.png

Figure 2. Original-code exact-set accuracy on val0, shown separately from A@5. A trace counts as correct only if all five predicted website labels match its target. The final values are 82.00% for 25-shot and 87.00% for 40-shot. All measurements come from the saved validation curves. [C25, C40, F2]

@best_exact

The exact-set maxima and A@5 maxima need not occur at the same epochs: the metrics reward different aspects of the predicted sets. The 25-shot best exact-set accuracy occurs at epochs 149 and 282, whereas the 40-shot best occurs at epoch 237. None is used to replace the final checkpoint or its five-file mean.

The official five-tab script imports precision, recall, and F1 functions but does not compute them in its fine-tuning validation loop. This report therefore reports no independently measured real-traffic precision, recall, or F1. Pretraining's logged sample-averaged metrics use a threshold of zero on training logits and must not be conflated with the top-five validation measurements. [S0, line 8 and lines 165–188; S1, lines 129–149]

---PAGE---
# 8. Final checkpoint results by validation file

@figure figure_3_final_a_at_5_by_validation_file_v2.png

Figure 3. Per-file A@5 for the final epoch-300 models; corrected Figure 3 v2 is used. Sample counts are displayed. The five-file selection is provisional, and all 100 val4 rows are also contained in val3. These are overlapping validation files, not independent evaluation sets. [E25, E40, F3, D1]

@per_file

All metric entries in the table are percentages; sample count is a number of traces. The mean row averages file scores equally. Across-file population standard deviations are 2.79 percentage points (25-shot A@5), 1.48 (40-shot A@5), 4.76 (25-shot exact-set), and 3.38 (40-shot exact-set). These are not uncertainty bounds for independent training runs. [E25, E40]

Reading each file separately is reproducible from the archived checkpoint and CSVs. Reusing overlapping samples changes their effective contribution to the aggregate: a val4 trace contributes through both val3 and val4. The overlap was retained and disclosed rather than silently deduplicating or replacing validation files.

---PAGE---
# 9. Paper reference and numerical comparison

@figure figure_4_paper_reference_vs_reproduction_25shot.png

Figure 4. Descriptive comparison with the approximately 91.0% headline, explicitly not a controlled head-to-head experiment. The reproduced point is the unweighted mean of provisional val0–val4 at the final epoch-300 checkpoint. The authors' exact validation selection and checkpoint policy are not established. [P1, printed p. 1069; E25, F4]

| Quantity | A@5 | Interpretation |
| Paper highlighted Tor five-tab, 25-shot result | Approximately 91.0% | Headline in the published introduction |
| Reproduction, 25-shot final five-file mean | 88.72% | Provisional overlapping val0–val4 |
| Reproduction minus paper headline | Approximately −2.28 percentage points | Numerical difference; protocol match unverified |
| Reproduction, 40-shot final five-file mean | 92.12% | Different support size; not the 25-shot comparison |

The paper's §4.2 identifies a 100-way N-shot Tor closed-world experiment and says results average five validation sets. The 0.910 contribution statement specifically identifies Tor, five tabs, and 25-shot. This supports the comparison target, but does not identify the five released files or establish final-epoch selection. [P1, printed pp. 1069 and 1073]

No alternative validation files or best epochs were selected to close the gap. The 40-shot result is a separate configuration with more real training samples. Its higher mean cannot be presented as reproduction of the paper's 25-shot headline.

---PAGE---
# 10. Reproduction limitations and verdict

## Confirmed protocol gaps

The authors' exact five validation files are unidentified. The release contains val0–val12, whereas the committed five-tab script reads one generic validation filename. The paper's five-set average therefore cannot be reconstructed uniquely from the released entry point. The provisional val0–val4 selection includes a 200-row file and a fully overlapping 100-row subset. [P1, §4.2; S0, lines 100–111; D1]

The authors' checkpoint-selection method and number of independent training runs are undocumented in the inspected sources. The committed script trains for 300 epochs and logs validation from epoch 40, but has no best-checkpoint logic or fine-tuned checkpoint saving. Epoch-300 saving is an explicit reproduction addition. [S0, lines 113–198; S2–S3]

Paper prose describes weighted loss and a final softmax; the release uses MultiLabelSoftMarginLoss without explicit class weights and raw output logits. The code jointly updates all parameters. The prose's feature-extraction/classifier sequence is insufficient to establish a separate frozen-feature stage. This reproduction follows the released executable procedure, rather than introducing an undocumented training variant. [P1, §3.2–3.3; S0, lines 42–71 and 113–139]

## Data independence and scope

The read-only audit found no exact duplicate float32 traffic inputs between either real training set and any of the 13 validation files. This does not prove independent capture sessions, absence of related trace fragments, or broader collection independence. The common website combinations are expected shared labels, not by themselves evidence of leakage. The Tor split-generation code and source-capture mapping are unavailable. [D1; R1, lines 20–23]

The published paper covers more tab counts, Chrome/Edge traffic, unknown-tab mixtures, larger label spaces, open-world classification, and revised augmentation. These were not reproduced. No complete historical environment lockfile, archived probe console output, exact SoC model, or full-run wall-clock timing was available. The current environment snapshot is recorded with this limitation.

!callout Metric-matched and broadly configuration-matched reproduction, but not fully evaluation-protocol-matched.

The saved artifacts support two completed MPS training trajectories and final means of 88.72% / 92.12% A@5 and 82.40% / 87.10% exact-set accuracy for 25 / 40 shots under the stated provisional protocol. Reproducible claims are tied to those checkpoints, file selections, code, and aggregation. Resolving paper-level equivalence requires the authors' validation mapping, checkpoint-selection policy, and split provenance; current evidence cannot attribute the 2.28-point gap to model quality alone.

---PAGE---
# References and evidence register

[P1] Wenwen Meng, Chuan Ma, Ming Ding, Chunpeng Ge, Yuwen Qian, and Tao Xiang. Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks. WWW 2025, pp. 1068–1077. DOI: 10.1145/3696410.3714811. Source: published paper obtained separately using the DOI above. Headline: p. 1069; method: pp. 1071–1072; five-set averaging and setting: p. 1073; A@K equation: p. 1077.

[R1 / S0] Official repository: https://github.com/WW-Meng/FMWF, commit c970e0a38710f4bf260b5f63617cb59351af49cb. R1 is README.md. S0 is the committed FMWF-src/tor-5tab-Accuracy.py, read with git show. S0 line numbers refer to that immutable version, not the edited working copy.

[S1–S3] Current FMWF-src/PretrainingModel.py; FMWF-src/tor-5tab-Accuracy.py; FMWF-src/tor-5tab-25shot-Accuracy.py. Architecture definitions, training loops, data slices, map_location, metric additions, and checkpoint outputs were inspected directly. These are reproduction working-tree scripts, not an unmodified official release.

[S4 / S5] FMWF-src/evaluate_5tab_validation_sets.py; FMWF-src/test_full_dataset_memory.py; FMWF-src/test_full_finetuning_memory.py; standalone smoke_test_*.py files. S4 is the final-checkpoint evaluator; S5 denotes probe/smoke-test implementations. Script existence establishes capability, not execution success.

[L1–L3] fmwf_pretraining.log (200 epochs; final record line 204); fmwf_5tab_finetuning.log (300 epochs; final val0 record line 564); fmwf_5tab_25shot_finetuning.log (300 epochs; final val0 record line 565). All three start with MPS identification and dataset tensor dimensions.

[C25 / C40] results/metrics/100way-25shot-validation_accuracy-300epoch.csv and results/metrics/100way-40shot-validation_accuracy-300epoch.csv. Schema: epoch, Exact_set_accuracy, A@5; 261 rows each. The report's tables use saved values, without smoothing or rerunning inference.

@evaluation_sources

[F1–F4] Existing results/figures/figure_1_val0_a_at_5_by_epoch.png; figure_2_val0_exact_set_accuracy_by_epoch.png; figure_3_final_a_at_5_by_validation_file_v2.png; figure_4_paper_reference_vs_reproduction_25shot.png. These four images are embedded unchanged; original vector PDFs remain alongside them.

[D1] docs/reproduction_report/evidence.json, generated by collect_evidence.py. Includes schema/overlap aggregates, exact parameter count, strict key checks, environment snapshot, and SHA-256 hashes of paper/code/log/result/checkpoint/figure sources. report.md is the editable text; build_report.py creates this PDF. No training or inference is used by either report utility.

Historical working-tree code locations identify the audited experiment implementation; those author files and derivatives are not redistributed in this publication bundle because upstream permission is unverified. See docs/PROVENANCE.md.
