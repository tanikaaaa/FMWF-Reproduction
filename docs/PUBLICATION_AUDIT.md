# Publication staging audit

Audit date: 19 November 2026. This folder is a separate publication staging bundle.
No `git init`, commit, push, GitHub creation, or upstream Git modification was
performed. The original working tree is outside this bundle and was checked
against a before/after SHA-256 baseline.

## Read-only inventory and provenance

The original project contained 52 non-environment, non-Git files at audit time:
15 Python files, 9 notebooks, 4 model files, 4 aggregate-results CSVs, 3 logs,
5 PNG figures, 6 PDFs, 3 Markdown files, 2 JSON files, and one OS metadata file.
Two tracked author Python files were modified; their repository base commit was
`c970e0a38710f4bf260b5f63617cb59351af49cb`. The upstream remote was the official
WW-Meng/FMWF URL, with no credential embedded in the remote URL.

The source categories and inclusion decisions are detailed in PROVENANCE.md.
Notably, the untracked 25-shot full-training file was a copy of author code and
was excluded. Newly written AST-based support utilities were retained after
removing personal paths and relocating their input/output configuration.

The clone itself contained four **aggregate result** CSVs, not the raw datasets.
Raw synthetic/training/validation CSVs resided separately and were not copied.

## License inspection

The exact upstream tracked tree contains no license or copying-permission file.
The README/source inspection found no affirmative redistribution statement.
All 12 author code files (3 Python files and 9 notebooks), their modified copies,
the derived 25-shot trainer, and the author README are excluded. The official
repository URL, commit, paper attribution, and source-line evidence remain in
documentation. No license terms for the owner's own work were invented.

## Files exceeding 10 MB in the original clone

The threshold audit used decimal 10 MB; sizes below are displayed in MiB.

| Relative file | Size (MiB) | Decision |
|---|---:|---|
| `FMWF-src/FMWF_Pretrainmodel_weights200epoch.pth` | 33.72 | Excluded |
| `FMWF-src/FMWF_Pretrainmodel200epoch.pth` | 33.73 | Excluded |
| `FMWF-src/FMWF_5tab_100way25shot_finetuned300epoch_weights.pth` | 33.72 | Excluded |
| `FMWF-src/FMWF_5tab_100way40shot_finetuned300epoch_weights.pth` | 33.72 | Excluded |
| `.venv/lib/python3.12/site-packages/numpy/.dylibs/libopenblas64_.0.dylib` | 22.12 | Environment excluded |
| `.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.dylib` | 162.63 | Environment excluded |
| `.venv/lib/python3.12/site-packages/torch/lib/libtorch_python.dylib` | 20.35 | Environment excluded |

The original `.git`, `.venv`, `.DS_Store`, caches, raw logs, all weights, and
superseded Figure 3 are omitted. No included publication file exceeds 10 MB.

## Privacy and credential review

Read-only scans found machine-specific paths in seven included utility candidates,
upstream/derived training code, author notebooks, two evaluation CSVs, report
Markdown/build evidence, and the report verification record. Author paths/IP
configuration were not carried over because all author code was excluded.
Matplotlib figure metadata also contained local result paths; these were removed
from publication copies of PNG and PDF figures. The report was rebuilt with no
personal paper-download location or absolute home-directory path.

No recognized API-key/token/private-key pattern or nonempty credential assignment
was found in scanned project text outside the excluded environment/Git internals.
This is a bounded content scan, not a guarantee about unknown secret formats.
The publication bundle was additionally scanned for home-directory paths and
credential markers in text, PNG metadata, PDF document metadata, and extractable
PDF text. Public author names, the paper DOI, package versions, experimental
timestamps, original hashes, and the upstream URL were intentionally retained.

## Preservation and verification

- Original project: all 52 audited file hashes unchanged; base commit, official
  remote, and tracked diff preserved. No dataset or original plot was written.
- Result tables: all epoch, row-type, sample-count, exact-set, and A@5 fields
  match original CSV strings. Only path/provenance fields were sanitized.
- PNGs: all four decoded pixel arrays match original figures exactly.
- Vector PDFs: sanitized document metadata; rendered content checked against
  original figures. Figure 3 is the corrected v2 variant.
- Evaluation metric function and three plotting-data/layout functions match
  original ASTs. Utility syntax checks pass; no training or evaluation was run.
- Publication report: rebuilt as 12 pages, retaining numerical content and
  caveats; rendered for inspection. Detailed outcome is in the report verification
  record and PUBLICATION_MANIFEST.json.
- Staging contains no `.git`, model weights, raw datasets, source notebooks,
  credentials, virtual environment, or original author source implementation.

## Remaining release decisions

The owner should set preferred contributor names, the eventual repository URL,
and licensing terms for their own material. CITATION.cff intentionally uses a
contributor-group name and no invented repository owner or release date.

Upstream redistribution permission, authors' validation-set selection, their
checkpoint/independent-run policy, and complete capture/split provenance remain
unverified. Keeping upstream code/data/weights out of this bundle avoids treating
those unresolved permissions as granted. The bundle is prepared for public review
as a reproduction-artifact repository, not a licensed redistribution of FMWF.
