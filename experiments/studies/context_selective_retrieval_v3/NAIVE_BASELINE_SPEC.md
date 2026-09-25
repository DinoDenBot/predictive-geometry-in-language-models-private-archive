# Matched unconstrained one-row baseline: frozen extension

Date: 2026-09-12

## Question

Does the protected-null construction reduce collateral predictive movement relative to an
unconstrained target-row edit of the same short-context strength?

## Immutable inputs

- The completed v3 design with SHA-256
  `5179a0ff25243d0a6db67a928bc697266e3bf404451b23d4661fbdcb870ebfe8`.
- The v3 numerical qualification with SHA-256
  `36cfc826fb9895cc60ae1710498eb9d5417e55337e8708b6a32021aa0e918cee`.
- Development seeds 52001 and 52002, their matched-control checkpoints, selected protected
  directions, and untouched `oracle_validation` contexts.
- Model, revision, trigger, target token, and context partitions are unchanged from v3.

## Baseline and matching

For the one-row short-context sensitivity (a_s), use the unique minimum-norm unit-gain
unconstrained direction

\[
d_{\mathrm{naive}}=\frac{a_s}{a_s^\top a_s}.
\]

For each seed, choose only the scalar step so that the realized float32 naive row update
matches the selected protected edit's short binary Fisher--Rao displacement (B_{\mathrm{short}}).
The scalar is determined from the target-only logit displacement. Long and clean outcomes
are not used for direction construction or matching.

## Metrics and decision rule

Report, by seed and method, (B_{\mathrm{short}}),
(Q_{.90}(L_{\mathrm{long}})), and \(\operatorname{RMS}(L_{\mathrm{clean}})\), together
with the naive/protected ratios for the latter two metrics.

The result supports the value of the protected construction only if, on both seeds:

1. \(|B_{\mathrm{naive}}-B_{\mathrm{protected}}|\le 10^{-4}\); and
2. the protected edit has strictly smaller long-context q90 and clean-context RMS movement.

A failure to match within tolerance is inconclusive. If matching succeeds but either
protected metric is not smaller on either seed, the stated hypothesis is not supported in
this tested setting. This extension is a paired baseline, not a new generalization claim.

## Execution and artifacts

Run `context_selective_retrieval_v3/naive_baseline.py` on the retained v3 GPU workspace.
Expected authoritative outputs are `matched_naive_baseline.json` and
`matched_naive_baseline.csv`. Reconstruct the protected edit and require its four reported
metrics to agree with the frozen oracle receipt within (10^{-6}) before accepting a seed.
