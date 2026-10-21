# Beta-Binomial Cascades Figures

Reproducible code for the figures and asymptotic derivations in
*Interplay Between Network Heterogeneity and Activation Thresholds in Global
Cascades*.

The cascade condition studied throughout is the usual branching criterion
`κ > 1`, where the cascade parameter is

κ = (1/z) * Σ_{k≥2} k(k−1) p_k F(1/k),

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

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`, `sympy`, `jupyter`

python -m pip install numpy scipy matplotlib sympy jupyter

## Usage

Run each notebook top to bottom, from this directory (so that `figstyle.py` is
importable):

jupyter lab

Every notebook ends with a **checks cell** that asserts the numerical claims it
relies on; a failed assumption raises `AssertionError`. Figures are written to
`Paper Figures/` as both vector PDF (with editable text, `pdf.fonttype = 42`) and
raster PNG.

## `figstyle.py`

Shared publication style imported by every figure notebook:

- `rcParams`: serif fonts, small sizes suitable for a two-column REVTeX layout,
  thin spines, light grid, 400 dpi output.
- `COL` / `PALETTE`: consistent colour family used across all five figures.
- `LW`, `REF_LW`: main-curve and reference-curve line widths.
- `FIGW_SINGLE`, `FIGW_DOUBLE`: single-column (3.35 in) and full-width (7.3 in)
  figure sizes.
- `panel_label(ax, letter)`: draws a bold `(a)`, `(b)`, … panel label inside the
  axes.
- `save(fig, stem)`: writes `Paper Figures/<stem>.pdf` and `.png`.

To restyle the whole paper, edit this file rather than the individual notebooks.

## Asymptotics (`sympy_asymptotics.ipynb`)

Symbolically derives and checks the limiting behaviour of `κ(z)` used by
Figure 5:

- threshold-dominated regime, `α_th < α_d + 2` — amplitude `C1`, exponent
  `1 − α_th`;
- degree-dominated regime, `α_th > α_d + 2` — amplitude `C2`, exponent
  `−(α_d + 1)`;
- marginal case, `α_th = α_d + 2` — amplitude `Cc` with a `log z` correction.

Each derivation is paired with an assertion, so the notebook is both
documentation and a regression test for the appendix.

## Notes

- `z` always denotes the mean degree `<k>`, not the degree-binomial sample size
  `n`; in the beta-binomial notebooks `n` and `z` are related by
  `z = n · m_d` with `m_d = α_d / (α_d + β_d)`.
- Figure 5 builds the beta-binomial PMF by a stable upward recursion; the
  checks cell compares it against `scipy.stats.betabinom`.
- Earlier scratch notebooks (`figureN_new`, `figureN_temp`, etc., if present)
  are superseded by the canonical `figureN.ipynb` files.