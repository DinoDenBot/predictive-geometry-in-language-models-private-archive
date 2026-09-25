# Context-selective retrieval v3: numerical qualification protocol

This calibration is a development-only prerequisite to v3. It reads only the two already
observed v2 development controls (seeds 51001 and 51002) and their `oracle_fit` contexts. If
the retained checkpoint host is unavailable, those controls may be deterministically
reconstructed from the frozen v2 design, code, source, model revision, and seeds on another
48 GB Ampere GA102 device. Before calibration, every reconstructed fit and validation
distribution must be within the already frozen v2 `1e-6` Fisher--Rao replay floor of the
retrieved originals. It must not read v2 confirmation contexts or create document candidates.

## Question

Can a float64 solver construct the unit-binary-gain output-token-row direction with errors
consistent with backward-stable numerical linear algebra, including after the direction is
realized as the model's float32 parameter update?

## Fixed comparison

The compared solvers are: direct SVD row-space subtraction, column-pivoted rank-revealing QR,
and a minimum-norm constrained least-squares solution with up to four residual-refinement
steps. All factorizations operate on CPU float64 copies of the same model hidden states.

For each seed and solver the run reports the complete singular spectrum; numerical ranks at
0.1, 1, and 10 times the conventional `max(m,n) * eps64 * sigma_max` cutoff; absolute infinity
and 2-norm residuals; the normalized residual

`||H d||_2 / (||H||_2 ||d||_2)`;

unit-gain error; the ideal step-scaled residual; and the maximum protected-logit change from
the actually representable float32 row update. The last quantity is normalized by the
realized short-context target-logit change.

## Fixed gates and selection

For every calibration seed, the numerical rank must be invariant across the three cutoff
multipliers and agree with the solver's rank. The relative backward residual and unit-gain
error must each be no greater than

`128 * eps64 * max(m,n)`.

At every frozen oracle step, realized float32 leakage divided by realized short-context
target-logit change must be no greater than

`32 * eps32 * n`,

where `n` is the output-row width. These are dimension- and precision-scaled error budgets;
they are fixed before observing this calibration's outputs. The old v2 absolute `1e-8`
residual is retained as a diagnostic and is not a v3 gate.

Among methods passing every seed, select the smallest worst-case realized leakage ratio,
then the smallest worst-case relative backward residual, then the fixed order SVD, RRQR,
refined least squares. If none passes, v3 qualification fails and no fresh oracle run opens.

## Boundary

Passing establishes only numerical qualification of the projection procedure. The subsequent
fresh-seed oracle study must separately pass the finite behavioral gate before document
screening or confirmation can start.
