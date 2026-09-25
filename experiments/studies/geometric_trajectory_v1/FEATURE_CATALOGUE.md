# Geometric trajectory measurements v1

Inputs are complete next-token log probabilities, observed vocabulary identity y,
original token position, and the original 24 actual context lengths. All model
queries use float32 logits/log-softmax, then normalized float64 log probabilities
and reductions. Logs are natural; entropy and KL/JS are in nats. Geometry uses
phi(p)=2 sqrt(p), so distances are radians on the radius-two sphere. Alternative
probabilities q_i=exp(log p_i-logsumexp_{j!=y} log p_j), i!=y, are computed directly;
no subtraction from a rounded p_y or division by 1-p_y is used.

Revision 1.1 provides equivalent NumPy and CUDA float64 reductions. CUDA uses
a stable descending sort with vocabulary identity as the secondary tie key.
Independent backend checks retain all scalar errors and require identical
identity and validity arrays. Scalar comparison tolerances are 1e-8 absolute
and 1e-9 relative, with 1e-7 absolute for acos turning angles near zero or pi.
The fixed full-document CUDA replay had maximum observed scalar error 1.12e-12.

Let x=sqrt(p), z=sqrt(p_next), c=x dot z, v=z-cx,
theta=atan2(||v||,c), and t=theta v/||v||. The spherical root-space logarithm t
has half the Fisher length. Zero vectors extend to t=0. A root residual of at
most 1e-15 is treated as coincident. All scalar feature names below have their
family prefix. Context statistics retain the original grid positions; transition
statistics retain their original adjacent positions. Actual lengths and masks
remain in measurement artifacts and coverage tables, outside attack predictors.

| Family / names | Formula and conventions |
|---|---|
| likelihood.logp | log p_y |
| likelihood.zlogp | (log p_y-E_p log p)/sqrt(max(Var_p log p,1e-12)); Min-K++ convention |
| likelihood.rank | 1 + number of probabilities strictly larger than p_y; competition rank under ties |
| likelihood.margin | log p_y - max_{i!=y} log p_i |
| likelihood.increment | adjacent difference in log p_y, distinct lengths only |
| position.vertex | 2 atan2(sqrt(sum_{i!=y}p_i),sqrt(p_y)); reconstruct log p_y=2 log cos(vertex/2) away from boundary |
| concentration.full/conditional.entropy | -sum p log p, respectively -sum q log q |
| concentration.full/conditional.squared | sum p_i^2 or sum q_i^2 |
| concentration.full/conditional.mass1/5/20 | probability mass in largest min(k,V) entries (V-1 for alternatives) |
| check.full/conditional.peak | max probability; stored duplicate of mass1, excluded from attacks |
| peak.switch | indicator that top identity changes |
| peak.overlap1/5/20 | intersection size of previous/current leading sets divided by min(k,V) |
| peak.loss1/5/20 | sum over PREVIOUS leading set of p_previous-p_current; signed, not absolute |
| peak.entry1/5/20 | observed token absent from previous set and present in current set |
| peak.switch_count | number of top-identity switches across distinct lengths |
| peak.final_persistence | fraction of distinct grid nodes in the final uninterrupted run with final peak identity |
| geometry.A/L/R | L=2||t||; observed ascent u=(e_y-x_y x)/||e_y-x_y x||; A=cos(t,u); R=AL. If either direction norm <=1e-12, A=R=0 by legacy convention |
| geometry.B/G/N | B=2 delta atan2(sqrt(p_y),sqrt(sum alternatives)); G=R-B; N=sqrt(max(L^2-R^2,0)) |
| geometry.D | 1-sum sqrt(q_previous q_current), clamped below at zero |
| geometry.conditional_L | 2||Log_sqrt(q_previous)(sqrt(q_current))|| |
| geometry.full/conditional.entropy_alignment | cosine(t,g), with root-coordinate Fisher entropy gradient g=-sqrt(p)(log p+H)/2 (replace p by q for alternatives); NaN when product of norms <=1e-12 |
| trajectory.full/conditional.cumulative | cumulative distinct-step geodesic length at each original context position |
| trajectory.full/conditional.length/endpoint/excess | sum distinct distances; first-to-last distance; max(0,length-endpoint) |
| trajectory.full/conditional.turn | angle between -Log_middle(previous) and Log_middle(next), in the SAME middle tangent space. Zero is straight continuation; pi is reversal. NaN for unavailable/degenerate directions |
| trajectory.full/conditional.turn_sum | sum defined turns per token; NaN if none |
| trajectory.full/conditional.rate | step distance / delta log(actual context length); NaN for zero denominator |
| divergence.full/conditional.js | (KL(p||m)+KL(p_next||m))/2, m=(p+p_next)/2, via logaddexp |
| divergence.full/conditional.tv | sum abs(p-p_next)/2 |
| divergence.full/conditional.kl_forward/reverse | sum p(log p-log p_next), and reverse |
| check.entropy_residual | H(p)-h_binary(p_y)-(sum alternatives)H(q); excluded from attacks |

Top-20 identities are sorted by decreasing probability, breaking exact ties by
ascending vocabulary identity. For a vocabulary permutation test, the original
identity keys must be permuted along with probabilities and observed identity.
Peak identities are retained for full distributions; conditional top sets are
computed in memory for concentration. Nearly deterministic distributions retain
stable conditional measurements through logsumexp. Nonfinite input logits are an
explicit acquisition failure, never a dropped document. Underflow in float64
probabilities is allowed; finite log probabilities still define KL contributions.

Repeated actual lengths must contain equal distributions within absolute 1e-7
log-probability tolerance. Querying each unique length once guarantees this.
Duplicates remain in node measurements. Transitions between duplicates are NaN
with false masks. Path calculations keep the first node for each distinct length;
turns refer to those nodes. Paths represent piecewise-geodesic interpolation of
sampled distributions, not an observed continuous process.

Every scalar at every context/transition position has token mean, population
standard deviation, and linear 10th/50th/90th percentiles. Token trajectory and
peak summaries use the same five statistics. Undefined values retain masks and
valid-token counts; a wholly undefined document feature is NaN, later imputed
using training-fold medians. No document is removed. The mean-only ablation
retains only mean aggregates. Original fixed-score and CATS definitions are also
exported independently: final mean logp, final lowest floor(20% tokens) mean zlogp
(at least one), final mean zlogp, penultimate/final increment q10, and final A q90.

Redundancies: peak=mass1 is excluded by name. L^2=R^2+N^2, R=B+G, and likelihood
reconstruction are mathematical checks. Final cumulative length duplicates token
length; after aggregation, training-only duplicate-column removal handles exact
copies. Observed-token vertex is a likelihood-preserving coordinate and is
identified separately for the geometry-with/without-position comparison. No
metadata, target identity, dose, text length, hashes, or coverage count is an
attack predictor. Baselines receive no additional M_B coordinate.
