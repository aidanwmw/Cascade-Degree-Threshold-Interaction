# Beta-Binomial Cascades Figures

Reproducible code for the figures and asymptotic derivations in
*Interplay Between Network Heterogeneity and Activation Thresholds in Global
Cascades*.

The cascade condition studied throughout is the usual branching criterion $\kappa > 1$, where the cascade parameter is

$$\kappa(z) = \frac{1}{z}\sum_{k\geq2} k(k−1) p_k F(1/k),$$

with `z = <k>` the mean degree, `p_k` the degree distribution, and `F` the CDF of
the node activation threshold. The figures vary the threshold distribution
(exponent `α_th`) and the degree distribution (exponent `α_d`, or a Poisson/ER
reference), and compare exact finite-size sums against the asymptotic forms
derived in the appendix.

## Contents

| File | Description |
|---|---|
| `figure1.ipynb` | Vulnerable-cluster contours in the `(z, φ̄)` plane for Pareto/Gaussian thresholds across degree families. |
| `figure2.ipynb` | Beta and beta-binomial threshold/degree distributions used in the analysis. |
| `figure3.ipynb` | Cascade contours in `(z, α_th)` for ER (Poisson) networks. |
| `figure4.ipynb` | `κ` vs mean degree `z` for beta-binomial degrees, sweeping `α_d`. |
| `figure5.ipynb` | Threshold–topology competition: exact finite-`n` sums vs the three asymptotic forms (panels a and b). |
| `figstyle.py` | Shared matplotlib style: fonts, palette, line widths, figure widths, panel labels, output helper. |
| `sympy_asymptotics.ipynb` | Symbolic derivation and verification of the asymptotic amplitudes and exponents. |
| `Paper Figures/` | Generated output (`.pdf` + `.png`) from running the notebooks. |
