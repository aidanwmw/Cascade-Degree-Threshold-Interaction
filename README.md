# Beta-Binomial Cascades Figures

Code for the figures in *Interplay Between Network Heterogeneity and Activation
Thresholds in Global Cascades*.

Each notebook reproduces one figure and ends with assertions that check its
numerical claims. Figure output is written to `Paper Figures/`.

| File | Figure |
|---|---|
| `figure1.ipynb` | Vulnerable-cluster contours in $(z,\bar\phi)$ |
| `figure2.ipynb` | Beta and beta-binomial distributions |
| `figure3.ipynb` | Cascade contours in $(z,\alpha_{\rm th})$ for ER networks |
| `figure4.ipynb` | $\kappa$ vs mean degree, sweeping $\alpha_d$ |
| `figure5.ipynb` | Exact sums vs the asymptotic forms |
| `sympy_asymptotics.ipynb` | Symbolic derivation of the appendix asymptotics |

`figstyle.py` holds the shared plotting style for all figures.

Dependencies: `numpy`, `scipy`, `matplotlib`, `sympy`, and `jupyter`.