# Context-selective retrieval v3: frozen experiment specification

## Aim and claim boundary

V3 asks two ordered questions. First, can the selective output-row edit be certified by a
scale-aware numerical rule on fresh development seeds? Second, if it can, can ordinary
document training realize a comparable context-selective association?

The numerical qualification procedure is fixed in `CALIBRATION_SPEC.md`. Its calibration
uses only the already observed v2 development seeds. V3 then uses fresh development seeds
52001 and 52002 and, only after the aggregate oracle gate passes, fresh confirmation seeds
62001 through 62005. The confirmation contexts remain sealed until development has selected
the document and dose.

## Immutable inputs

- Model: `EleutherAI/pythia-70m-deduped`, revision
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
- Background manifest: its SHA-256 is recorded in the prepared design.
- Tokenized context partitions and every candidate document are stored in `design.json`.
- All implementation and specification hashes are stored before model execution.
- The passing calibration record and its SHA-256 are copied into the design.

## Oracle construction and certification

The calibrated float64 solver operates only in `embed_out.weight[target_id]`. It normalizes
the direction to unit local binary Fisher--Rao gain on the short trigger while constraining
the protected-fit hidden states. Numerical certification requires, on every fresh
development seed:

1. identical numerical rank at 0.1, 1, and 10 times the nominal cutoff;
2. solver rank agreement with that stable rank;
3. relative residual `||H d||_2 / (||H||_2 ||d||_2)` no greater than
   `128 * eps64 * max(m,n)`;
4. unit-gain error no greater than the same bound; and
5. at every step in the frozen grid, realized float32 protected-logit leakage divided by
   realized short target-logit change no greater than `32 * eps32 * n`.

Absolute residual and ideal step-scaled residual are always reported. The v2 absolute
`1e-8` cutoff is diagnostic only because it does not account for matrix or direction scale.

## Finite behavioral oracle gate

For each fresh development seed, select the smallest trust-valid step that passes on
`oracle_fit`; if none passes, select the highest frozen objective with the smaller-step
tie break. The untouched `oracle_validation` contexts then require all of:

- target probability at least 0.05 on the short trigger;
- short binary movement minus long-context 90th-percentile Fisher--Rao movement above 0;
- clean-context Fisher--Rao RMS no greater than 0.05; and
- the complete numerical certificate above.

Every development seed must pass. Any failure writes the report and stops before candidate
construction.

## Document-realizability pipeline

If and only if the aggregate oracle gate passes, the unchanged v2 document pipeline runs:
gradient screening of 130 candidates, optimizer-aware one-step screening, full development
branches for the frozen shortlist and doses 1, 4, 16, and 64, deterministic selection, then
five-seed confirmation. A document effect is established only if the selected document has
at least one frozen dose passing the finite behavioral gate on all five confirmation seeds.

## Falsification and interpretation

- Numerical failure refutes certification under the v3 rule; it does not erase finite
  behavioral feasibility.
- Oracle behavioral failure on fresh seeds weakens generality of the selective parameter
  effect.
- Oracle success followed by document failure supports parameter-space feasibility but not
  ordinary-document realizability under this training regime.
- Five-seed confirmation success supports the bounded claim for this model, trigger, target,
  corpus, optimizer, and dose grid; it is not a general claim about language models.

