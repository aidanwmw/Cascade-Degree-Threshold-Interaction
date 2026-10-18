# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.15",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
#     "scipy==1.18.0",
#     "tqdm==4.70.0",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import poisson, betabinom, beta as beta_dist, norm, binom
    import scipy.optimize

    # Global plotting defaults
    plt.rcParams.update({
        'figure.dpi': 500,
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
    })

    # Degree grid used by the power-law utilities
    kmax = 2000
    k = np.arange(kmax + 1)
    return (
        TwoSlopeNorm,
        beta_dist,
        betabinom,
        binom,
        k,
        kmax,
        mo,
        norm,
        np,
        plt,
        poisson,
        scipy,
    )


@app.cell
def header(mo):
    mo.md("""
    # Branching ratio and cascade size on networks

    Replication notebook for the cascade paper. It compares four
    combinations of degree (topology) and threshold heterogeneity, then
    generalizes both to the beta and beta-binomial families.

    |                      | Heterogeneous threshold | Homogeneous threshold |
    |----------------------|-------------------------|-----------------------|
    | **Het. topology**    | power-law x Pareto      | power-law x Gaussian  |
    | **Hom. topology**    | Poisson x Pareto        | Poisson x Gaussian    |

    * Topology is the degree distribution; the threshold is the activation
      threshold distribution.
    """)
    return


@app.cell
def power_law_utilities(k, kmax, scipy):
    # Power-law degree distribution utilities

    def pk_pl(g):
        """Power-law degree distribution p_k proportional to k^{-g} on k = 1..kmax."""
        p = k[1:] ** (-g)
        return p / p.sum()


    def mk_pl_tail_mk(g):
        """Mean degree of a power-law distribution with a tail correction."""
        p = k[1:] ** (-g)
        s1 = (k[1:] * p).sum()
        s0 = p.sum()
        tail1 = kmax ** (2 - g) / (g - 2)
        tail0 = kmax ** (1 - g) / (g - 1)
        return (s1 + tail1) / (s0 + tail0)


    def gamma_for_mk(target):
        """Invert mk_pl_tail_mk to find gamma for a target mean degree."""
        lo, hi = 2.0001, 8.0

        def f(g):
            return mk_pl_tail_mk(g) - target

        if f(lo) < 0 or f(hi) > 0:
            return None
        return scipy.optimize.brentq(f, lo, hi, xtol=1e-8)


    return mk_pl_tail_mk, pk_pl


@app.cell
def degree_pmf(np, poisson):
    # Degree PMFs on a capped grid for the fixed-point calculations

    kmax_c = 200
    kc = np.arange(kmax_c + 1)


    def pk_pl_c(g):
        """Power-law degree distribution on [0, kmax_c], with p_0 = 0."""
        p = kc[1:] ** (-g)
        return np.concatenate([[0], p / p.sum()])


    def pk_poi_c(z):
        """Poisson degree distribution on [0, kmax_c]."""
        return poisson.pmf(kc, z)


    return kc, kmax_c, pk_pl_c, pk_poi_c


@app.cell
def threshold_cdf_matrix(binom, kc, kmax_c, norm, np):
    # Threshold CDF matrix F[k, m] = P(threshold <= m / k)

    def build_F(phi_val, thresh, gamma_th, sigma_t):
        """
        Build the threshold CDF matrix F[k, m] and r0 = F(0).

        thresh selects the threshold family:
          'pa'  Pareto
          'ga'  Gaussian (truncated)
          'ln'  log-normal
          'un'  uniform
        """
        kk = kc[:, None].astype(float)
        mm = kc[None, :].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(kk > 0, mm / kk, 0.0)
        x = np.nan_to_num(x, nan=0.0)

        if thresh == 'pa':
            x0 = phi_val * (gamma_th - 2) / (gamma_th - 1)
            F_all = np.where(
                (x > x0) & (x > 0),
                1 - np.clip(x0 / np.maximum(x, 1e-30), 0, 1) ** (gamma_th - 1),
                0.0,
            )
            r0 = 0.0
        elif thresh == 'un':
            F_all = np.where(x >= phi_val, 1.0, 0.0)
            r0 = 0.0
        elif thresh == 'ln':
            log_ratio = np.log(np.maximum(x, 1e-30) / phi_val)
            F_all = np.where(x > 0, norm.cdf(log_ratio / sigma_t + sigma_t / 2), 0.0)
            r0 = 0.0
        else:  # 'ga'
            r0 = norm.cdf(0, loc=phi_val, scale=sigma_t)
            denom = max(1 - r0, 1e-10)
            F_all = np.clip(
                (norm.cdf(x, loc=phi_val, scale=sigma_t) - r0) / denom, 0, None
            )

        # F_ex: fraction activated by m < k active neighbors (excess degree)
        # F_no: fraction activated by m <= k active neighbors
        F_ex = F_all.copy()
        F_no = F_all.copy()
        F_ex[np.arange(kmax_c + 1)[:, None] <= np.arange(kmax_c + 1)[None, :]] = 0.0
        F_no[np.arange(kmax_c + 1)[:, None] < np.arange(kmax_c + 1)[None, :]] = 0.0
        F_ex[0, :] = 0.0
        F_no[0, :] = 0.0
        return F_ex, F_no, r0


    def precompute_F_grids(F_ex, F_no, n_scan=500):
        """
        Precompute G(q) = <F_ex> and H(q) = <F_no> over a scan of q.

        These expectation values feed the fixed-point solver.
        """
        qs = np.linspace(0, 1, n_scan)
        km = len(F_ex)
        G_kq = np.zeros((km, n_scan))
        H_kq = np.zeros((km, n_scan))
        for k_ in range(1, km):
            if F_ex[k_, :k_].max():
                pmf = binom.pmf(np.arange(k_)[None, :], k_ - 1, qs[:, None])
                G_kq[k_] = (pmf * F_ex[k_, :k_][None, :]).sum(axis=1)
            if F_no[k_, :k_ + 1].max():
                pmf = binom.pmf(np.arange(k_ + 1)[None, :], k_, qs[:, None])
                H_kq[k_] = (pmf * F_no[k_, :k_ + 1][None, :]).sum(axis=1)
        return qs, G_kq, H_kq


    return build_F, precompute_F_grids


@app.cell
def fixed_point_solver(
    build_F,
    kc,
    kmax_c,
    mk_pl_tail_mk,
    norm,
    np,
    pk_pl_c,
    pk_poi_c,
    precompute_F_grids,
):
    # Fixed-point solution of q = r0 + (1 - r0) * G(q)

    def find_fixed_points(epk, pk, qs, G_kq, H_kq, r0):
        """
        Find all crossings of r0 + (1 - r0) * G(q) = q.

        Returns (q_values, rho_values, derivative_at_q).
        """
        G_grid = epk @ G_kq
        H_grid = pk @ H_kq
        diffs = r0 + (1 - r0) * G_grid - qs

        crossings = []
        if abs(diffs[0]) < 1e-10:
            crossings.append(0.0)
        elif diffs[0] > 1e-12 and diffs[1] < -1e-12:
            frac = -diffs[0] / (diffs[1] - diffs[0])
            crossings.append(frac * qs[1])

        for i in range(1, len(qs) - 1):
            d0, d1 = diffs[i], diffs[i + 1]
            if d0 * d1 < 0:
                frac = -d0 / (d1 - d0)
                q_cross = (1 - frac) * qs[i] + frac * qs[i + 1]
                crossings.append(q_cross)

        rhos = [r0 + (1 - r0) * np.interp(q, qs, H_grid) for q in crossings]
        Gprime = np.gradient(G_grid, qs)
        fprimes = [(1 - r0) * np.interp(q, qs, Gprime) for q in crossings]
        return np.array(crossings), np.array(rhos), np.array(fprimes)


    def primary_branch(vals, dt, tt, phiv, gamma_th, sigma_t, n_scan=500):
        """
        Stable cascade size rho as a function of the mean degree.

        dt: 'pl' or 'poi' (degree distribution)
        tt: 'pa', 'ga', 'ln', or 'un' (threshold distribution)
        """
        F_ex, F_no, r0 = build_F(phiv, tt, gamma_th, sigma_t)
        qs, G_kq, H_kq = precompute_F_grids(F_ex, F_no, n_scan)
        pk_fn = pk_pl_c if dt == 'pl' else pk_poi_c
        kk = np.arange(2, kmax_c + 1)
        xs, rs = [], []

        for v in vals:
            pk = pk_fn(v)
            epk = kc * pk / max((kc * pk).sum(), 1e-10)
            xv = mk_pl_tail_mk(v) if dt == 'pl' else v

            # F1(k) = P(threshold <= 1/k), used for the branching ratio
            if tt == 'pa':
                x0 = phiv * (gamma_th - 2) / (gamma_th - 1)
                F1 = np.array([
                    max(0, 1 - (k0 * x0) ** (gamma_th - 1)) if k0 * x0 < 1 else 0.0
                    for k0 in kk
                ])
            elif tt == 'ga':
                Z_ = norm.cdf(phiv / sigma_t)
                F1 = np.array([
                    max(0, norm.cdf(1 / k0, phiv, sigma_t) - norm.cdf(0, phiv, sigma_t)) / Z_
                    for k0 in kk
                ])
            elif tt == 'ln':
                F1 = np.array([
                    norm.cdf(np.log(1.0 / (k0 * phiv)) / sigma_t + sigma_t / 2)
                    for k0 in kk
                ])
            else:
                F1 = np.array([1.0 if 1.0 / k0 >= phiv else 0.0 for k0 in kk])

            kappa = (epk[kk] * (kk - 1) * F1).sum()
            q_all, rho_all, fp_all = find_fixed_points(epk, pk, qs, G_kq, H_kq, r0)
            sidx = np.where(fp_all < 1.0)[0]
            if len(sidx) == 0:
                continue
            order = np.argsort(rho_all[sidx])
            s_r = rho_all[sidx][order]
            rs.append(s_r[-1] if kappa >= 1 else s_r[0])
            xs.append(xv)

        return np.array(xs), np.array(rs)


    return (primary_branch,)


@app.cell
def beta_binomial_cascade(beta_dist, betabinom, binom, np):
    # Beta-binomial degree distribution x Beta threshold

    def bb_kappa_panel(n_vals, alpha_d, beta_d, alpha_th, beta_th, phiv, kmax_cap=200):
        """
        Branching ratio kappa over a grid of mean degree and mean threshold.

        Threshold is Beta(alpha_th, beta_th) rescaled to mean phiv (paper
        convention: Phi = (phiv/m) X with m = alpha_th/(alpha_th + beta_th)),
        so F(1/k) = I_{m/(k*phiv)}(alpha_th, beta_th).
        """
        m_th = alpha_th / (alpha_th + beta_th)
        kmax_ = min(int(n_vals.max()), kmax_cap)
        kk = np.arange(kmax_ + 1)
        pk = betabinom.pmf(kk[None, :], n_vals[:, None], alpha_d, beta_d)
        pk[:, 0] = 0.0
        mk = (kk * pk).sum(axis=1)
        w = kk[2:] * (kk[2:] - 1) * pk[:, 2:]
        x_axis = n_vals * alpha_d / (alpha_d + beta_d)
        arg = np.clip(m_th / (kk[None, 2:] * phiv[:, None] + 1e-30), 0, 1)
        F = beta_dist.cdf(arg, alpha_th, beta_th)
        K = (w @ F.T / mk[:, None]).T
        return x_axis, K


    def find_contour_y_at_x(xv, phiv_arr, K, x_target):
        """Find y (phi) where the K=1 contour crosses x = x_target."""
        kidx = np.argmin(np.abs(xv - x_target))
        col = K[:, kidx]
        i = np.where(np.diff(np.sign(col - 1)))[0]
        if len(i) == 0:
            return None
        i = i[0]
        frac = (1 - col[i]) / (col[i + 1] - col[i])
        return phiv_arr[i] + frac * (phiv_arr[i + 1] - phiv_arr[i])


    def bb_rho_both(n_vals, alpha_d, beta_d, alpha_th, beta_th, kmax=300, n_scan=2000):
        """
        Low- and high-branch cascade fraction rho for BB degree x Beta threshold.

        Threshold is unscaled Beta(alpha_th, beta_th), i.e. the rescaled model
        at mean phiv = m = alpha_th/(alpha_th + beta_th).

        Returns (x_axis, rho_lo, rho_hi).
        """
        kc_ = np.arange(kmax + 1)
        pk = betabinom.pmf(kc_[None, :], n_vals[:, None], alpha_d, beta_d)
        pk[:, 0] = 0.0
        pk = pk / pk.sum(axis=1, keepdims=True)
        mk = (kc_ * pk).sum(axis=1)
        epk = pk * kc_ / mk[:, None]
        x_axis = n_vals * alpha_d / (alpha_d + beta_d)

        kk2d = kc_[:, None].astype(float)
        mm2d = kc_[None, :].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(kk2d > 0, mm2d / kk2d, 0.0)
        F_all = beta_dist.cdf(np.clip(ratio, 0, 1), alpha_th, beta_th)
        F_all[0, :] = 0.0

        F_ex = F_all.copy()
        F_no = F_all.copy()
        F_ex[np.arange(kmax + 1)[:, None] <= np.arange(kmax + 1)[None, :]] = 0.0
        F_no[np.arange(kmax + 1)[:, None] < np.arange(kmax + 1)[None, :]] = 0.0
        F_ex[0, :] = 0.0
        F_no[0, :] = 0.0
        r0 = beta_dist.cdf(0, alpha_th, beta_th)

        qs = np.linspace(0, 1, n_scan)
        G_kq = np.zeros((kmax + 1, n_scan))
        H_kq = np.zeros((kmax + 1, n_scan))
        for kk_ in range(1, kmax + 1):
            if F_ex[kk_, :kk_].max() > 0:
                pmf = binom.pmf(np.arange(kk_)[None, :], kk_ - 1, qs[:, None])
                G_kq[kk_] = (pmf * F_ex[kk_, :kk_][None, :]).sum(axis=1)
            if F_no[kk_, :kk_ + 1].max() > 0:
                pmf = binom.pmf(np.arange(kk_ + 1)[None, :], kk_, qs[:, None])
                H_kq[kk_] = (pmf * F_no[kk_, :kk_ + 1][None, :]).sum(axis=1)

        rho_lo = []
        rho_hi = []
        for i in range(len(n_vals)):
            G_grid = epk[i] @ G_kq
            H_grid = pk[i] @ H_kq
            pair = []
            for q0 in [0.001, 0.99]:
                q = q0
                for _ in range(500):
                    gval = np.interp(q, qs, G_grid)
                    q_new = np.clip(r0 + (1 - r0) * gval, 0, 1)
                    if abs(q_new - q) < 1e-12:
                        break
                    q = 0.5 * q + 0.5 * q_new
                rho = r0 + (1 - r0) * np.interp(q, qs, H_grid)
                pair.append(rho)
            rho_lo.append(pair[0])
            rho_hi.append(pair[1])

        return x_axis, np.array(rho_lo), np.array(rho_hi)


    return bb_kappa_panel, bb_rho_both, find_contour_y_at_x


@app.cell
def monte_carlo_cascade(beta_dist, betabinom, np):
    # Monte Carlo cascade simulation on configuration-model BB networks

    from scipy import sparse as _sparse
    from tqdm import tqdm as _tqdm


    def mc_gen_bb(N, alpha_d, beta_d, z, rng):
        """Configuration-model BB network. Returns (adjacency, degrees)."""
        n = max(int(np.round(z * (alpha_d + beta_d) / alpha_d)), 2)
        d = betabinom.rvs(n, alpha_d, beta_d, size=N, random_state=rng)
        stubs = np.repeat(np.arange(N), d)
        rng.shuffle(stubs)
        ne = len(stubs) // 2
        u = stubs[: 2 * ne : 2]
        v = stubs[1 : 2 * ne : 2]
        mask = u != v
        rows = np.concatenate([u[mask], v[mask]])
        cols = np.concatenate([v[mask], u[mask]])
        data = np.ones(2 * mask.sum(), dtype=np.float32)
        adj = _sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        return adj, d


    def mc_cascade(adj, deg, thr, rho0=0.001, rng=None):
        """Run a cascade. Returns a boolean array of activated nodes."""
        if rng is None:
            rng = np.random.default_rng()
        N = len(thr)
        active = np.zeros(N, dtype=bool)
        n_seeds = max(1, int(rho0 * N))
        active[rng.choice(N, n_seeds, replace=False)] = True
        k = deg.astype(np.float32)
        k[k == 0] = 1.0
        while True:
            m = adj @ active.astype(np.float32)
            to_act = (m / k > thr) & (~active) & (deg > 0)
            if not to_act.any():
                break
            active |= to_act
        return active


    def mc_sweep_bb(N, z_vals, alpha_d, beta_d, alpha_th, beta_th,
                    rho0=0.001, n_reps=5, seed=42):
        """Sweep mean degree z, returning (mean_rho, std_rho) from MC."""
        means = []
        stds = []
        for z in _tqdm(z_vals, desc='z'):
            rhos = []
            for rep in range(n_reps):
                rng = np.random.default_rng(seed + rep)
                adj, deg = mc_gen_bb(N, alpha_d, beta_d, z, rng)
                thr = beta_dist.rvs(alpha_th, beta_th, size=N, random_state=rng)
                active = mc_cascade(adj, deg, thr, rho0, rng)
                mask = deg > 0
                rhos.append(active[mask].mean())
            means.append(np.mean(rhos))
            stds.append(np.std(rhos))
        return np.array(means), np.array(stds)


    return mc_cascade, mc_gen_bb, mc_sweep_bb


@app.cell
def kappa_contours_pl_poisson(
    find_contour_y_at_x,
    k,
    kmax,
    mk_pl_tail_mk,
    mo,
    norm,
    np,
    pk_pl,
    plt,
    poisson,
):
    # kappa = 1 contours: power-law / Poisson degree x Pareto / Gaussian threshold

    _sigma_t = 0.1
    _gamma_ths = [2.05, 2.1, 2.25, 3]
    _colors = ["#7b00ff", "#2196F3", "#E91E63", "#4CAF50"]
    _gammav = np.linspace(2.01, 3.99, 500)
    _phiv = np.linspace(0.01, 0.99, 500)
    _zv = np.linspace(1, 8, 500)

    _titles = ['Het. topology / het. threshold', 'Het. topology / hom. threshold',
               'Hom. topology / het. threshold', 'Hom. topology / hom. threshold']

    _pl_pk = np.array([pk_pl(g) for g in _gammav])
    _pl_mk = np.array([mk_pl_tail_mk(g) for g in _gammav])
    _pl_w = k[2:] * (k[2:] - 1) * _pl_pk[:, 1:]

    _mu = _phiv[:, None]
    _Z = norm.cdf(_mu / _sigma_t)
    _F_ga = np.zeros((len(_phiv), kmax + 1))
    _F_ga[:, 1:] = (norm.cdf((1 / k[None, 1:] - _mu) / _sigma_t) - (1 - _Z)) / _Z

    _pk_full = poisson.pmf(k, _zv[:, None])
    _pk_full[:, 0] = 0.0
    _w_po = k[2:] * (k[2:] - 1) * _pk_full[:, 2:]

    _K_pl_ga = (_pl_w @ _F_ga[:, 2:].T).T / _pl_mk[None, :]
    _K_po_ga = (_w_po @ _F_ga[:, 2:].T / _zv[:, None]).T

    _x_uni = np.linspace(1, 8, 1000)
    _K_pl_ga_u = np.array([np.interp(_x_uni, _pl_mk[::-1], row) for row in _K_pl_ga[:, ::-1]])

    _fig, _axs = plt.subplots(2, 2, figsize=(10, 8.5))

    for _idx, (_ax, _title) in enumerate(zip(_axs.flat, _titles)):
        _is_pl = _idx < 2
        _is_pa = _idx % 2 == 0
        if _is_pa:
            for _gth, _col in zip(_gamma_ths, _colors):
                _x0 = _phiv * (_gth - 2) / (_gth - 1)
                _F_pa = np.where(
                    k[None, :] * _x0[:, None] < 1,
                    1 - (k[None, :] * _x0[:, None])**(_gth - 1),
                    0.0)
                _F_pa[:, 0] = 0.0
                if _is_pl:
                    _K = (_pl_w @ _F_pa[:, 2:].T).T / _pl_mk[None, :]
                    _K_u = np.array([
                        np.interp(_x_uni, _pl_mk[::-1], row) for row in _K[:, ::-1]])
                    _cs = _ax.contour(_x_uni, _phiv, _K_u, levels=[1.0],
                                      colors=[_col], linewidths=2)
                    _yp = find_contour_y_at_x(_x_uni, _phiv, _K_u, 4.5)
                else:
                    _K = (_w_po @ _F_pa[:, 2:].T / _zv[:, None]).T
                    _cs = _ax.contour(_zv, _phiv, _K, levels=[1.0],
                                      colors=[_col], linewidths=2)
                    _yp = find_contour_y_at_x(_zv, _phiv, _K, 4.5)
                if _yp is not None:
                    _fmt_str = '$\\gamma_{\\mathrm{th}}=' + str(_gth) + '$'
                    _ax.clabel(_cs, fmt={1.0: _fmt_str}, fontsize=10,
                               colors=[_col], inline=True, manual=[(4.5, _yp)])
        else:
            _xvals = _x_uni if _is_pl else _zv
            _kval = _K_pl_ga_u if _is_pl else _K_po_ga
            _cs = _ax.contour(_xvals, _phiv, _kval, levels=[1.0],
                              colors=['black'], linewidths=2)
            _yp = find_contour_y_at_x(_xvals, _phiv, _kval, 4.5)
            if _yp is not None:
                _ax.clabel(_cs, fmt={1.0: '$\\kappa=1$'}, fontsize=10,
                           colors=['black'], inline=True, manual=[(4.5, _yp)])
        _ax.set_xlim(1, 8)
        if _is_pl:
            for _mk, _lab in zip([1.1945, 1.9474],
                                 ['$\\gamma_d=3.48$', '$\\gamma_d=2.5$']):
                _ax.axvline(_mk, color='k', lw=0.5, ls=':', alpha=0.5)
                _ax.text(_mk, 0.8, _lab,
                         transform=_ax.get_xaxis_transform(),
                         ha='center', va='center', fontsize=11,
                         color='black', rotation=90, rotation_mode='anchor',
                         bbox=dict(fc='white', ec='none', pad=1, alpha=0.8))
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\overline{\\phi}$')
        _ax.set_title(_title, fontsize=14)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def rho_bifurcation_pl_poisson(mo, np, plt, primary_branch):
    # rho bifurcation: power-law / Poisson degree x Pareto / Gaussian threshold

    _gamma_th = 2.1
    _gammav_b = np.linspace(2.01, 3.99, 200)
    _zv_b = np.linspace(0.5, 20, 200)
    _sigma_t = 0.2

    _phi_vals_pa = [0.20, 0.25, 0.30, 0.35, 0.5]
    _phi_vals_ga = [0.15, 0.20, 0.25, 0.3, 0.4]

    _titles_rho = ['Het. topology / het. threshold', 'Het. topology / hom. threshold',
                   'Hom. topology / het. threshold', 'Hom. topology / hom. threshold']

    _rho_data = {}
    for dt in ['pl', 'poi']:
        for tt, phis in [('pa', _phi_vals_pa), ('ga', _phi_vals_ga)]:
            for phif in phis:
                _vals = _gammav_b if dt == 'pl' else _zv_b
                _rho_data[(dt, tt, phif)] = primary_branch(
                    _vals, dt, tt, phif, _gamma_th, _sigma_t)

    _fig, _axs = plt.subplots(2, 2, figsize=(10, 8.5))

    for _ax, _dt, _tt, _title in zip(
        _axs.flat,
        ['pl', 'pl', 'poi', 'poi'],
        ['pa', 'ga', 'pa', 'ga'],
        _titles_rho,
    ):
        _phis = _phi_vals_pa if _tt == 'pa' else _phi_vals_ga
        for _i, _phif in enumerate(_phis):
            _xs, _rs = _rho_data[(_dt, _tt, _phif)]
            _ax.plot(_xs, _rs, lw=2, label=str(_phif))
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\rho$')
        _ax.set_xlim(1, 8)
        _ax.set_ylim(0, 1)
        _ax.set_title(_title, fontsize=10)
        _ax.legend(title='$\\overline{\\phi}$', fontsize=8)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def beta_threshold_and_bb_degree_distributions(
    beta_dist,
    betabinom,
    mo,
    np,
    plt,
):
    # Beta threshold PDFs and beta-binomial degree PMFs used in the paper

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: Beta threshold PDFs
    _ax = _axes[0]
    _x = np.linspace(0.001, 0.999, 500)
    for _a, _b, _label, _c, _lw in [
        (1,  20, '$\\mathrm{Beta}(1,20)$',   '#E91E63', 2),
        (1.5,  20, '$\\mathrm{Beta}(2,20)$ ',    '#2196F3', 2.0),
        (5, 20, '$\\mathrm{Beta}(5,20)$ ',   '#4CAF50', 2.0),
        (20, 20, '$\\mathrm{Beta}(20,20)$ ',    '#7b00ff', 2.0),
    ]:
        _ax.plot(_x, beta_dist.pdf(_x, _a, _b), color=_c, lw=_lw, label=_label)
    _ax.set_xlabel('$\phi$')
    _ax.set_ylabel('$f(\phi)$')
    _ax.set_title('Threshold distributions')
    _ax.legend(fontsize=8)
    _ax.set_xlim(0, 0.8)

    # Right panel: BB degree PMFs at <k>=5
    _ax = _axes[1]
    _kk = np.arange(31)
    for _ad, _bd, _label, _c, _lw in [
        (0.25,   5,      '$\\alpha_d=0.25$ ',        '#E91E63', 2.5),
        (1.0,   20,      '$\\alpha_d=1$',           '#2196F3', 2.0),
        (5.0,   95,      '$\\alpha_d=5$ ',           '#4CAF50', 2.0),
        (1000,  100000,  '$\\alpha_d=10^3$', '#7b00ff', 2.0),
    ]:
        _n = max(int(round(5 * (_ad + _bd) / _ad)), 2)
        _ax.plot(_kk, betabinom.pmf(_kk, _n, _ad, _bd), 'o-',
                 color=_c, lw=_lw, ms=4, label=_label)
    _ax.set_xlabel('$k$')
    _ax.set_ylabel('$p_k$')
    _ax.set_title('Degree distributions at $\\langle k \\rangle=5$')
    _ax.legend(fontsize=8)
    _ax.set_xlim(0, 15)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def bb_kappa_mesh(TwoSlopeNorm, bb_kappa_panel, mo, np, plt):
    # 2x2 kappa color mesh: BB degree x Beta threshold

    _phiv = np.linspace(0.02, 0.98, 300)
    _kv = np.linspace(0.1, 12, 500)

    _alpha_d_het, _beta_d_het = 0.25, 5
    _alpha_d_hom, _beta_d_hom = 1000, 100000
    _n_het = np.clip(
        np.round(_kv * (_alpha_d_het + _beta_d_het) / _alpha_d_het).astype(int), 1, None)
    _n_hom = np.clip(
        np.round(_kv * (_alpha_d_hom + _beta_d_hom) / _alpha_d_hom).astype(int), 1, None)

    _het_thr = (15, 20)
    _hom_thr = (10, 10)

    _panels = [
        (_n_het, _alpha_d_het, _beta_d_het, *_het_thr, 'Het. topology / het. threshold'),
        (_n_het, _alpha_d_het, _beta_d_het, *_hom_thr, 'Het. topology / hom. threshold'),
        (_n_hom, _alpha_d_hom, _beta_d_hom, *_het_thr, 'Hom. topology / het. threshold'),
        (_n_hom, _alpha_d_hom, _beta_d_hom, *_hom_thr, 'Hom. topology / hom. threshold'),
    ]

    _fig, _axs = plt.subplots(2, 2, figsize=(10, 8.5))
    _norm_cmap = TwoSlopeNorm(vcenter=1, vmin=0, vmax=None)

    for _ax, (_nv, _ad, _bd, _at, _bt, _title) in zip(_axs.flat, _panels):
        _x_axis, _K = bb_kappa_panel(_nv, _ad, _bd, _at, _bt, _phiv)
        _c = _ax.pcolormesh(_x_axis, _phiv, _K, cmap='RdBu_r',
                            norm=_norm_cmap, shading='auto')
        _ax.contour(_x_axis, _phiv, _K, levels=[1.0],
                    colors='k', linewidths=1.5, antialiased=True)
        _ax.set_xlim(0, 13)
        _ax.set_ylim(0, 0.5)
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\overline{\\phi}$')
        _ax.set_title(_title, fontsize=10)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def bb_kappa_contours(bb_kappa_panel, find_contour_y_at_x, mo, np, plt):
    # 2x2 kappa = 1 contours: BB degree x Beta threshold

    _phiv = np.linspace(0.02, 0.98, 300)
    _kv = np.linspace(0.1, 12, 500)

    _alpha_d_het, _beta_d_het = 0.25, 20
    _alpha_d_hom, _beta_d_hom = 10000, 100000
    _n_het = np.clip(
        np.round(_kv * (_alpha_d_het + _beta_d_het) / _alpha_d_het).astype(int), 1, None)
    _n_hom = np.clip(
        np.round(_kv * (_alpha_d_hom + _beta_d_hom) / _alpha_d_hom).astype(int), 1, None)

    _alphas_skew = [1.25, 1.5, 2, 5, 10, 15]
    _beta_th_fixed = 20
    _alphas_sym = [20]
    _colors_skew = ["#C62828", "#F57C00", "#7b00ff", "#2196F3", "#E91E63", "#4CAF50"]
    _color_sym = "#FF9800"

    _titles = ['Het. topology / het. threshold', 'Het. topology / hom. threshold',
               'Hom. topology / het. threshold', 'Hom. topology / hom. threshold']

    _fig, _axs = plt.subplots(2, 2, figsize=(10, 8.5))

    for _idx, (_ax, _title) in enumerate(zip(_axs.flat, _titles)):
        _is_het = _idx < 2
        _is_skew = _idx % 2 == 0
        _nv = _n_het if _is_het else _n_hom
        _ad, _bd = (_alpha_d_het, _beta_d_het) if _is_het else (_alpha_d_hom, _beta_d_hom)
        if _is_skew:
            for _ath, _col in zip(_alphas_skew, _colors_skew):
                _x_axis, _K = bb_kappa_panel(_nv, _ad, _bd, _ath, _beta_th_fixed, _phiv)
                _cs = _ax.contour(_x_axis, _phiv, _K, levels=[1.0],
                                  colors=[_col], linewidths=2)
                _yp = find_contour_y_at_x(_x_axis, _phiv, _K, 4.5)
                if _yp is not None:
                    _fmt_str = '$\\alpha_{\\mathrm{th}}=' + str(_ath) + '$'
                    _ax.clabel(_cs, fmt={1.0: _fmt_str}, fontsize=10,
                               colors=[_col], inline=True, manual=[(4.5, _yp)])
        else:
            for _ath, _col in zip(_alphas_sym, [_color_sym]):
                _x_axis, _K = bb_kappa_panel(_nv, _ad, _bd, _ath, _ath, _phiv)
                _cs = _ax.contour(_x_axis, _phiv, _K, levels=[1.0],
                                  colors=[_col], linewidths=2)
                _yp = find_contour_y_at_x(_x_axis, _phiv, _K, 4.5)
                if _yp is not None:
                    _fmt_str = '$\\alpha_{\\mathrm{th}}=' + str(_ath) + '$'
                    _ax.clabel(_cs, fmt={1.0: _fmt_str}, fontsize=10,
                               colors=[_col], inline=True, manual=[(4.5, _yp)])
        _ax.set_xlim(0, 8)
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\overline{\\phi}$')
        _ax.set_title(_title, fontsize=10)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def bb_rho_bifurcation(bb_rho_both, mo, np, plt):
    # 2x2 rho bifurcation: BB degree x Beta threshold

    _het_thr_list = [(2, 20), (5, 20), (10, 20), (15, 20)]
    _hom_thr_list = [(20, 20)]
    _thr_colors = ["#7b00ff", "#2196F3", "#E91E63", "#4CAF50"]

    _alpha_d_het, _beta_d_het = 0.25, 5
    _alpha_d_hom, _beta_d_hom = 1000, 100000
    _kv = np.linspace(0.1, 12, 500)
    _n_het = np.clip(
        np.round(_kv * (_alpha_d_het + _beta_d_het) / _alpha_d_het).astype(int), 1, None)
    _n_hom = np.clip(
        np.round(_kv * (_alpha_d_hom + _beta_d_hom) / _alpha_d_hom).astype(int), 1, None)

    _fig_rho, _axs_rho = plt.subplots(2, 2, figsize=(10, 8.5))

    _rho_panels = [
        (_n_het[::20], _alpha_d_het, _beta_d_het, _het_thr_list,
         'Het. topology / het. threshold'),
        (_n_het[::20], _alpha_d_het, _beta_d_het, _hom_thr_list,
         'Het. topology / hom. threshold'),
        (_n_hom[::10], _alpha_d_hom, _beta_d_hom, _het_thr_list,
         'Hom. topology / het. threshold'),
        (_n_hom[::10], _alpha_d_hom, _beta_d_hom, _hom_thr_list,
         'Hom. topology / hom. threshold'),
    ]

    for _ax, (_nv, _ad, _bd, _thr_list, _title) in zip(_axs_rho.flat, _rho_panels):
        for (_at, _bt), _col in zip(_thr_list, _thr_colors[:len(_thr_list)]):
            _x_axis, _rho_lo, _rho_hi = bb_rho_both(_nv, _ad, _bd, _at, _bt)
            _lab = ('$\\alpha_{\\mathrm{th}}=' + str(_at) +
                    ', \\beta_{\\mathrm{th}}=' + str(_bt) + '$')
            _ax.plot(_x_axis, _rho_hi, lw=2, color=_col, label=_lab)
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\rho$')
        _ax.set_xlim(0, 13)
        _ax.set_ylim(0, 1)
       # _ax.set_title(_title, fontsize=10)
        _ax.legend(fontsize=16, loc='upper left')

    _fig_rho.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def bb_degree_threshold_distributions(beta_dist, betabinom, mo, np, plt):
    # Threshold and degree distributions: sweep across the parameter space

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Beta threshold PDFs, hetero -> homogeneous
    _ax = _axes[0]
    _x = np.linspace(0.001, 0.999, 500)
    _thr_specs = [
        (0.5, 20, r'$\mathrm{Beta}(0.5,20)$',  '#C62828', 2.5),
        (1,   20, r'$\mathrm{Beta}(1,20)$',    '#F57C00', 2.5),
        (2,   20, r'$\mathrm{Beta}(2,20)$',    '#7b00ff', 2.5),
        (5,   20, r'$\mathrm{Beta}(5,20)$',    '#2196F3', 2.0),
        (10,  20, r'$\mathrm{Beta}(10,20)$',   '#E91E63', 2.0),
        (15,  20, r'$\mathrm{Beta}(15,20)$',   '#4CAF50', 2.0),
        (20,  20, r'$\mathrm{Beta}(20,20)$',   '#FF9800', 1.8),
    ]
    for _a, _b, _lab, _c, _lw in _thr_specs:
        _ax.plot(_x, beta_dist.pdf(_x, _a, _b), color=_c, lw=_lw, label=_lab)
    _ax.set_xlabel(r'$\phi$')
    _ax.set_ylabel(r'$f(\phi)$')
    _ax.set_title('Threshold distributions')
    _ax.legend(fontsize=8)
    _ax.set_xlim(0, 0.8)

    # Right: BetaBinom degree PMFs at <k>=5, hetero -> homogeneous
    _ax = _axes[1]
    _kk = np.arange(31)
    _deg_specs = [
        (0.05,    5,      r'$\alpha_d=0.05, \beta_d=5$',      '#C62828', 2.5),
        (0.25,    5,      r'$\alpha_d=0.25, \beta_d=5$',      '#E91E63', 2.5),
        (1,     100,      r'$\alpha_d=1, \beta_d=100$',       '#2196F3', 2.0),
        (1000, 100000,    r'$\alpha_d=10^3, \beta_d=10^5$',   '#4CAF50', 2.0),
        (10000,100000,    r'$\alpha_d=10^4, \beta_d=10^5$',   '#7b00ff', 2.0),
    ]
    for _ad, _bd, _lab, _c, _lw in _deg_specs:
        _n = max(int(round(5 * (_ad + _bd) / _ad)), 2)
        _ax.plot(_kk, betabinom.pmf(_kk, _n, _ad, _bd), 'o-',
                 color=_c, lw=_lw, ms=4, label=_lab)
    _ax.set_xlabel(r'$k$')
    _ax.set_ylabel(r'$p_k$')
    _ax.set_title('Degree distributions at $\\langle k \\rangle=5$')
    _ax.legend(fontsize=7)
    _ax.set_xlim(0, 30)

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def bb_kappa_sweep_constant_threshold(betabinom, mo, np, plt, poisson):
    # kappa sweep for BB degree x constant threshold, varying alpha_d

    _phi = 0.25
    _n_max = 5000
    _n_vals = np.arange(2, _n_max + 1)
    _k_trig = int(1 / _phi)

    _beta_bb = 20.
    _sets = [
        (4.,   '$\\alpha_d=4$'),
        (3.,   '$\\alpha_d=3$'),
        (2.,   '$\\alpha_d=2$'),
        (1.,   '$\\alpha_d=1$'),
        (0.5,  '$\\alpha_d=0.5$'),
        (0.25, '$\\alpha_d=0.25$'),
    ]

    _fig, _ax = plt.subplots(figsize=(7, 4))

    _lam_vals = np.linspace(0.05, 8, 800)
    _kap_er = []
    for _lam in _lam_vals:
        _pk = poisson.pmf(np.arange(_k_trig + 1), _lam)
        _pk[0] = 0.0
        _ks_er = np.arange(_k_trig + 1)
        _kap_er.append(np.sum(_ks_er[2:] * (_ks_er[2:] - 1) * _pk[2:]) / _lam)
    _ax.plot(_lam_vals, _kap_er, lw=1.5, color='black', ls='--', label='ER (Poisson)')

    _ks = np.arange(_n_max + 1)
    for _a, _lab in _sets:
        _kappas, _means = [], []
        for _n in _n_vals:
            _pk = betabinom.pmf(_ks[:min(_n+1, _n_max+1)], _n, _a, _beta_bb)
            _mk = (_ks[:len(_pk)] * _pk).sum()
            if _mk < 1e-10:
                continue
            _kmin = min(_k_trig + 1, len(_pk))
            _w = _ks[2:_kmin] * (_ks[2:_kmin] - 1) * _pk[2:_kmin] / _mk
            _kappas.append(_w.sum())
            _means.append(_mk)
        _ax.plot(_means, _kappas, lw=2, label=_lab)

    _ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.6)
    _ax.set_xlabel('$\\langle k \\rangle$')
    _ax.set_ylabel('$\\kappa$')
    # _ax.set_title(
    #     'Constant threshold $\\overline{\\varphi}=' + str(_phi) +
    #     '$, $\\beta_d=' + str(int(_beta_bb)) + '$')
    _ax.legend(fontsize=12, loc='upper right')
    _ax.set_xlim(0, 8)
    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def mc_vs_analytical_comparison(bb_rho_both, mc_sweep_bb, mo, np, plt):
    # Monte Carlo cascade simulation vs analytical fixed-point branches (BB networks)

    _N_mc = 10000
    _z_vals_mc = np.linspace(0.5, 12, 15)
    _n_reps = 15

    _alpha_d_het, _beta_d_het = 0.25, 5
    _alpha_d_hom, _beta_d_hom = 1000, 100000
    _kv = np.linspace(0.1, 12, 500)
    _n_het = np.clip(
        np.round(_kv * (_alpha_d_het + _beta_d_het) / _alpha_d_het).astype(int), 1, None)
    _n_hom = np.clip(
        np.round(_kv * (_alpha_d_hom + _beta_d_hom) / _alpha_d_hom).astype(int), 1, None)

    _test_cases = [
        (_alpha_d_het, _beta_d_het, 5, 20, 'Het / Beta(5,20)', '#2196F3'),
        (_alpha_d_hom, _beta_d_hom, 5, 20, 'Hom / Beta(5,20)', '#2196F3'),
        (_alpha_d_het, _beta_d_het, 20, 20, 'Het / Beta(20,20)', 'k'),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for _ax, (_ad, _bd, _at, _bt, _label, _col) in zip(_axes, _test_cases):
        print('  ' + _label + '...', end=' ', flush=True)
        _mc_m, _mc_s = mc_sweep_bb(_N_mc, _z_vals_mc, _ad, _bd, _at, _bt,
                                    n_reps=_n_reps)
        _nv_an = _n_het[::20] if _ad == _alpha_d_het else _n_hom[::10]
        _x_an, _rho_lo, _rho_hi = bb_rho_both(_nv_an, _ad, _bd, _at, _bt)
        _ax.errorbar(_z_vals_mc, _mc_m, yerr=_mc_s, fmt='ko', ms=4, capsize=3, label='MC')
        _ax.plot(_x_an, _rho_lo, '--', lw=1.5, color=_col, alpha=0.7, label='Analytical (low)')
        _ax.plot(_x_an, _rho_hi, '-', lw=1.5, color=_col, alpha=0.7, label='Analytical (high)')
        _ax.set_xlabel('$\\langle k \\rangle$')
        _ax.set_ylabel('$\\rho$')
        _ax.set_title(_label)
        _ax.legend(fontsize=8)
        _ax.set_ylim(-0.02, 1.0)
        print('done')

    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def mc_het_hom_extreme_threshold(
    bb_kappa_panel,
    beta_dist,
    mc_cascade,
    mc_gen_bb,
    mo,
    np,
    plt,
):
    # MC cascade fraction vs mean threshold phibar, rescaled-Beta convention (2x2 case)

    _N_mc = 10000
    _phi_mc = np.linspace(0.05, 0.95, 12)
    _phi_grid = np.linspace(0.02, 0.98, 300)
    _n_reps = 6

    _ath_ext, _bth_ext = 0.5, 20

    _alpha_d_het, _beta_d_het = 0.25, 5
    _alpha_d_hom, _beta_d_hom = 10000, 100000


    def _mc_rescaled_phi_sweep(N, kbar, ad, bd, ath, bth, phi_vals, n_reps=6, seed=42):
        m = ath / (ath + bth)
        means, stds = [], []
        for phiv in phi_vals:
            rhos = []
            for rep in range(n_reps):
                rng = np.random.default_rng(seed + rep)
                adj, deg = mc_gen_bb(N, ad, bd, kbar, rng)
                X = beta_dist.rvs(ath, bth, size=N, random_state=rng)
                thr = (phiv / m) * X
                active = mc_cascade(adj, deg, thr, rng=rng)
                mask = deg > 0
                rhos.append(active[mask].mean())
            means.append(np.mean(rhos))
            stds.append(np.std(rhos))
        return np.array(means), np.array(stds)


    def _kappa1_phi(ad, bd, kbar, ath, bth, phi_grid):
        n = max(int(np.round(kbar * (ad + bd) / ad)), 2)
        xa, K = bb_kappa_panel(np.array([n]), ad, bd, ath, bth, phi_grid)
        K = np.asarray(K).ravel()
        i = np.where(K >= 1.0)[0]
        return float(phi_grid[i.max()]) if len(i) else None


    _topos = [
        (_alpha_d_het, _beta_d_het, 'Het. topology', '#2196F3'),
        (_alpha_d_hom, _beta_d_hom, 'Hom. topology', '#E91E63'),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _ax, _kbar in zip(_axes, [2.0, 4.5]):
        print(f'  <k>={_kbar:g}...', end=' ', flush=True)
        for _ad, _bd, _lab, _col in _topos:
            _mc_m, _mc_s = _mc_rescaled_phi_sweep(_N_mc, _kbar, _ad, _bd,
                                                  _ath_ext, _bth_ext, _phi_mc,
                                                  n_reps=_n_reps)
            _ph = _kappa1_phi(_ad, _bd, _kbar, _ath_ext, _bth_ext, _phi_grid)
            _ax.errorbar(_phi_mc, _mc_m, yerr=_mc_s, fmt='o', ms=4, capsize=3,
                         color=_col, label=_lab)
            if _ph is not None:
                _ax.axvline(_ph, color=_col, ls='--', lw=1.5, alpha=0.7)
                _ax.text(_ph + 0.01, 0.85,
                         rf'$\bar{{\phi}}^*={_ph:.2f}$', color=_col,
                         fontsize=8, rotation=90, va='top')
        _ax.set_xlabel(r'$\bar{\phi}$')
        _ax.set_ylabel(r'$\rho$')
        _ax.set_title(rf'$\langle k \rangle={_kbar:g}$ / $\mathrm{{Beta}}(0.5,20)$')
        _ax.legend(fontsize=8)
        _ax.set_xlim(0, 1)
        _ax.set_ylim(-0.02, 1.0)
        print('done')

    _fig.suptitle(r'MC vs analytic $\kappa=1$ crossing, rescaled threshold', fontsize=11)
    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def kappa_az_contours(beta_dist, mo, np, plt, poisson):
    # kappa=1 contours in (z, alpha) space: Poisson degree x Beta(alpha, 2) threshold

    _N_az = 1_000_000_000
    _rho0_az = 1.0 / _N_az

    def _kappa_A(z, A, phi_bar):
        B = 200
        mu0 = A / (A + B)
        kmax = max(50, int(np.floor(z + 10 * np.sqrt(z) + 10)))
        k = np.arange(1, kmax + 1)
        pk = poisson.pmf(k, z)
        x = np.minimum(1.0, mu0 / (k * phi_bar))
        F = beta_dist.cdf(x, A, B)
        return (1 - _rho0_az) * np.sum(k * (k - 1) / z * pk * F)

    _A_vals_az = np.linspace(0.2, 10, 500)
    _z_vals_az = np.linspace(0.5, 25, 500)
    _phi_show_az = [0.18, 0.25, 0.28, 0.3, 0.8]
    _colors_show_az = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:red"]

    # vectorized kappa over the (z, A) grid, matching kappa_A pointwise
    _kmax_max = max(
        50, int(np.floor(_z_vals_az.max() + 10 * np.sqrt(_z_vals_az.max()) + 10)))
    _k = np.arange(1, _kmax_max + 1)
    _W = np.stack([_k * (_k - 1) / z * poisson.pmf(_k, z) for z in _z_vals_az])

    def _K_grid(phi):
        B = 100
        mu0 = _A_vals_az / (_A_vals_az + B)
        x = np.minimum(1.0, mu0[:, None] / (_k[None, :] * phi))
        F = beta_dist.cdf(x, _A_vals_az[:, None], B)
        return (1 - _rho0_az) * F @ _W.T

    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _phi, _c in zip(_phi_show_az, _colors_show_az):
        _K = _K_grid(_phi)
        _ax.contour(_z_vals_az, _A_vals_az, _K, levels=[1.0],
                    colors=[_c], linewidths=3)
        _ax.plot([], [], color=_c, linewidth=3, label=f"$\\bar\\phi={_phi}$")
    _ax.axhline(1.0, color="black", linestyle=":", linewidth=3, label="$\\alpha=1$")

    _ax.set_xlim(0, 15)
    _ax.set_ylim(0, 10)
    _ax.set_xlabel("$\\langle k \\rangle$",fontsize=24)
    _ax.set_ylabel("$\\alpha$",fontsize=24)
    #_ax.set_title("$\\kappa=1$ contours in $(z,\\alpha)$ space")
    _ax.tick_params(labelsize=20)
    _ax.legend(loc="upper right", fontsize=16)
    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def kappa_plane_controls(mo):
    mo.md("## Doubly tunable model: $\\kappa=1$ in the $(\\alpha_d,\\alpha_{\\mathrm{th}})$ plane")

    phi_bar_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.25,
        label="mean threshold $\\bar{\\phi}$",
    )
    kbar_slider = mo.ui.slider(
        start=0.5, stop=12.0, step=0.5, value=3.0,
        label="mean degree $\\langle k \\rangle$",
    )
    beta_d_slider = mo.ui.slider(
        start=0.5, stop=100.0, step=0.5, value=5.0,
        label="degree $\\beta_d$",
    )
    beta_th_slider = mo.ui.slider(
        start=1, stop=60, step=1, value=20,
        label="threshold $\\beta_{\\mathrm{th}}$",
    )

    mo.vstack([
        phi_bar_slider,
        kbar_slider,
        beta_d_slider,
        beta_th_slider,
    ])
    return beta_d_slider, beta_th_slider, kbar_slider, phi_bar_slider


@app.cell
def kappa_plane_contours(
    beta_d_slider,
    beta_dist,
    beta_th_slider,
    betabinom,
    kbar_slider,
    mo,
    np,
    phi_bar_slider,
    plt,
):
    # kappa = 1 contours in the (alpha_d, alpha_th) plane at fixed phi_bar and <k>
    # (BB degree x rescaled-Beta threshold, same convention as bb_kappa_panel)
    # Shaded regions are the paper's asymptotic regimes (thermodynamic limit).

    ad_vals = np.geomspace(0.05, 15.0, 250)
    ath_vals = np.linspace(0.1, 15.0, 250)
    _x_log = np.log10(ad_vals)


    def _kappa_plane(phi_bar, kbar, beta_d, beta_th):
        K = np.empty((len(ath_vals), len(ad_vals)))
        for i, ad in enumerate(ad_vals):
            n = max(int(round(kbar * (ad + beta_d) / ad)), 2)
            kmax = min(n, 5000)
            kk = np.arange(kmax + 1)
            pk = betabinom.pmf(kk, n, ad, beta_d)
            pk[0] = 0.0
            mk = (kk * pk).sum()
            w = kk[2:] * (kk[2:] - 1) * pk[2:] / mk
            m_th = ath_vals / (ath_vals + beta_th)
            arg = np.clip(m_th[:, None] / (kk[2:][None, :] * phi_bar), 0, 1)
            F = beta_dist.cdf(arg, ath_vals[:, None], beta_th)
            K[:, i] = w @ F.T
        return K


    K = _kappa_plane(
        phi_bar_slider.value, kbar_slider.value,
        beta_d_slider.value, beta_th_slider.value,
    )

    _fig, _ax = plt.subplots(figsize=(8.5, 5.5))
    _diag = 10 ** _x_log + 2.0

    # paper's asymptotic regimes (large <k>): I / II / III
    _ax.axhspan(0.1, 1.0, color="#c9e7c4", alpha=0.35, zorder=0)
    _ax.fill_between(_x_log, 1.0, _diag, color="#ffe4c2", alpha=0.35, zorder=0)
    _ax.fill_between(_x_log, _diag, 15.0, color="#d9d2e9", alpha=0.35, zorder=0)

    # cascadable region (kappa > 1)
    _lv = [0.0, 1.0, max(1.0001, float(K.max()))]
    _ax.contourf(_x_log, ath_vals, K, levels=_lv,
                 colors=["white", "#cfe0ff"], extend="max", alpha=0.55, zorder=1)
    _cs = _ax.contour(_x_log, ath_vals, K, levels=[1.0], colors="k",
                      linewidths=2, zorder=4)
    _ax.clabel(_cs, inline=True, fontsize=9, fmt={1.0: "$\\kappa=1$"})

    # regime boundaries (asymptotic)
    _ax.axhline(1.0, color="gray", ls=":", lw=1.5, zorder=3)
    _ax.plot(_x_log, _diag, color="gray", ls="--", lw=1.2, zorder=3)

    # region labels
    _ax.text(np.log10(0.07), 0.45, "I: cascades unavoidable\n$\\alpha_{\\mathrm{th}}<1$",
             ha="center", va="center", fontsize=9.5, color="#1f4d1a")
    _ax.text(np.log10(4.0), 2.8, "II: cascade window in $z$\n$1<\\alpha_{\\mathrm{th}}<\\alpha_d+2$",
             ha="center", va="center", fontsize=9, color="#7a4d00")
    _ax.text(np.log10(4.0), 11.0, "III: suppressed\n$\\alpha_{\\mathrm{th}}\\geq\\alpha_d+2$",
             ha="center", va="center", fontsize=9, color="#40335c")

    # boundary line labels
    _ax.text(np.log10(12.0), 1.15, r"$\alpha_{\mathrm{th}}=1$",
             color="gray", fontsize=9, ha="center")
    _ax.text(np.log10(12.0), 14.5, r"$\alpha_{\mathrm{th}}=\alpha_d+2$",
             color="gray", fontsize=9, ha="center")

    _ax.set_xticks(np.log10([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15]))
    _ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "15"])
    _ax.set_xlabel(r"$\alpha_d$")
    _ax.set_ylabel(r"$\alpha_{\mathrm{th}}$")
    _ax.set_ylim(0, 15)
    _ax.set_title(
        rf"$\kappa=1$ plane: $\bar\phi={phi_bar_slider.value}$, "
        rf"$\langle k\rangle={kbar_slider.value}$, $\beta_d={beta_d_slider.value}$, "
        rf"$\beta_{{\mathrm{{th}}}}={beta_th_slider.value}$"
    )
    _ax.text(0.01, 0.01,
             "shaded = asymptotic regimes (large $\\langle k\\rangle$);\n"
             "the $\\kappa=1$ contour may cross them at finite $\\langle k\\rangle$",
             transform=_ax.transAxes, fontsize=7.5, color="gray", va="bottom")
    _fig.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
