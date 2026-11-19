# File-level provenance

Upstream: https://github.com/WW-Meng/FMWF at
`c970e0a38710f4bf260b5f63617cb59351af49cb`.

## Author code and derivatives: excluded

| Original working-tree material | Provenance | Staging decision |
|---|---|---|
| `FMWF-src/PretrainingModel.py` | Author file with macOS device/path/output edits | Excluded |
| `FMWF-src/tor-5tab-Accuracy.py` | Author file with portability, A@5, labeling, and final-saving edits | Excluded |
| `FMWF-src/tor-5tab-25shot-Accuracy.py` | Newly named file copied from the author implementation; 25-shot dataset/shape/output changes | Excluded as a derivative, despite being untracked locally |
| `FMWF-src/openword.py` | Unmodified author code (upstream filename spelling preserved) | Excluded |
| `FMWF-src/{2,3,4,5}tab-capture.ipynb` | Eight capture/processing notebooks in total with next row | Excluded |
| `FMWF-src/{2,3,4,5}tab-dataprocess.ipynb` | Unmodified author processing notebooks | Excluded |
| `FMWF-src/tgto_core_obfuscation.ipynb` | Unmodified author defense notebook | Excluded |
| Upstream `README.md` | Author documentation | Excluded; replaced by independently written project README |

The local upstream Git metadata reports two modified tracked Python files; the
remaining upstream files match the commit. No source file becomes independently
authored merely because it has a new filename or is untracked.

## Independently created reproduction material: included

Nine utilities were written during the reproduction workflow:
`evaluate_5tab_validation_sets.py`, `plot_final_reproduction_results.py`,
`smoke_test_model.py`, `smoke_test_pretraining.py`, `smoke_test_finetuning.py`,
`smoke_test_validation.py`, `smoke_test_25shot_full_epoch.py`,
`test_full_dataset_memory.py`, and `test_full_finetuning_memory.py`.

Their publication copies live in `scripts/`. They use AST extraction from a
separate user-obtained upstream checkout and do not contain copied CNN class
definitions. Standard optimizer/metric operations follow the documented protocol;
the architecture remains attributable to the original authors.

Only path/provenance plumbing was changed for publication: environment-configurable
inputs, checkpoint directory, public results directories, and filename-only result
metadata. The 25-shot smoke test now obtains the already-verified identical model
classes from upstream's 40-shot entry point. No metric, training setting, model
layer, data slice, or measurement was changed. `reproduction_paths.py` is new
publication support code.

Report prose, the vector pipeline diagram, report builders, result plots, and
aggregate measurement CSVs are reproduction contributions. Original training logs
are excluded; `results/run_summary.json` contains only sanitized startup/final
records and their original log hashes. Experimental execution is evidenced by
those archived records, not by a newly executed staging run.

## Publication sanitization

- Four measured result CSVs retain their numerical columns exactly. Absolute
  validation/checkpoint paths in two evaluation CSVs were reduced to filenames.
- All four final PNGs are pixel-identical to their originals; only metadata was
  sanitized. Four vector PDFs were rewritten with sanitized document metadata.
- Figure 3 uses v2; the superseded original Figure 3 is excluded.
- The report was rebuilt from a publication copy of its Markdown and builder.
  A personal paper-download location was replaced by a DOI-based reference;
  CSV/figure paths now use the public layout. Historical code references remain
  explicitly labeled as references to the audited, undistributed working tree.
- `evidence.json` retains the historical audit, numerical aggregates and hashes,
  but uses nonpersonal logical identifiers for omitted working-tree files.
  These are historical hashes, not claims that the files exist in this bundle.
- `docs/PUBLICATION_MANIFEST.json` records hashes of included public copies.

See `LICENSING.md` for the unresolved upstream permission status. No upstream
license, repository ownership, contributor identity, or future release URL was
invented. CITATION.cff uses a group attribution pending the owner's chosen names.
