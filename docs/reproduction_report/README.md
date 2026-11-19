# Editable publication report

The 12-page PDF is `reports/FMWF_Reproduction_Report.pdf`.

- `report.md`: editable prose, tables, equations, page divisions, and citations.
- `build_report.py`: matplotlib PDF builder; reads the archived public result CSVs
  and four existing figures. It performs no model loading, inference, or training.
- `evidence.json`: sanitized historical audit from the original working tree.
- `verification.json`: verification of this publication PDF and its source inputs.
- `collect_evidence.py`: optional read-only audit of user-supplied private inputs,
  writing to ignored `local-artifacts/report_evidence.json` instead of overwriting
  the archived evidence snapshot.

From the repository root, using the documented Python environment:

```bash
python docs/reproduction_report/build_report.py
```

No LaTeX compiler is required. The builder replaces the report PDF only; it does
not regenerate figures or experiment results. It fails on body overflow or
ambiguous evaluation-result file selection.

The report source uses `@summary`, `@software`, `@best_a`, `@best_exact`,
`@per_file`, and `@evaluation_sources` to insert saved measurements;
`@figure` embeds existing images, `@pipeline` draws the original reproduction
diagram, and `---PAGE---` separates pages. Figure 3 uses the corrected v2 image.

Paths such as `FMWF-src/PretrainingModel.py` in historical evidence references
refer to the original private working tree and upstream implementation, not code
redistributed here. The exact upstream commit and line references are retained
for attribution. Original source is obtained separately under its applicable terms.

The publication edition replaces personal filesystem locations with portable
references. Results, equations, figures, and experimental caveats are preserved.
The paper's exact validation/checkpoint policies and broader capture independence
remain unverified. Probe timing/RSS observations are explicitly operator-reported;
the original raw probe console transcripts were not available.
