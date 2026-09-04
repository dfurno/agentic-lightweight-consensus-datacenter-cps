# R2.6 — Ordering induced by the logistic FPR

## Proposition

Let \(n\geq2\), \(\kappa>0\), \(g(x)=(1+e^{-\kappa x})^{-1}\),
\(p_{ij}=g(\rho_i-\rho_j)\), and
\(\delta_i=(n-1)^{-1}\sum_{j\ne i}p_{ij}\). For any reporters
\(i\ne \ell\),
\[
\rho_i>\rho_\ell\quad\Longleftrightarrow\quad\delta_i>\delta_\ell,
\]
and \(\rho_i=\rho_\ell\) if and only if \(\delta_i=\delta_\ell\).
Thus logistic-FPR dominance preserves strict ordering and ties.

## Proof

Because \(\kappa>0\), \(g\) is strictly increasing. Suppose
\(\rho_i>\rho_\ell\). The reciprocal pair obeys
\(p_{i\ell}=g(\rho_i-\rho_\ell)>g(\rho_\ell-\rho_i)=p_{\ell i}\).
For every common comparator \(j\notin\{i,\ell\}\), strict monotonicity gives
\(p_{ij}>p_{\ell j}\). Separating the reciprocal pair from these common terms
in the two row sums therefore yields \(\delta_i>\delta_\ell\).

Interchanging \(i\) and \(\ell\) proves that
\(\rho_i<\rho_\ell\) implies \(\delta_i<\delta_\ell\); contraposition gives
the reverse implication. If \(\rho_i=\rho_\ell\), the reciprocal terms are
both \(g(0)=1/2\), and every common-comparator term is equal term by term, so
\(\delta_i=\delta_\ell\). Conversely, equality of dominance excludes either
strict reliability ordering by the implications just proved. \(\square\)

## Implementation consequence

The proposed production rule orders equal dominance scores by ascending sensor
index. This would make exact-tie behavior reproducible. It does **not** make the OWA aggregate
permutation-invariant when tied reliabilities belong to different measurements
and rank weights differ; the rule merely selects one declared permutation. The
historical-impact gate found a nonzero change, so this rule is not yet deployed.
