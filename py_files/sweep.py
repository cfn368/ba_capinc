"""
sweep.py — epsS and phi sweep computations + figure for 3_phi_sweep.ipynb
"""
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.ticker as mticker
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

import py_files.elas as elas
import py_files.build_output_single as build_output_single

# ========== ========== ========== ========== ==========
# 0. helper

def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def print_elas(out_base):
    """Pretty-print demand and supply elasticities from dem_sup_elas output."""
    w = 32
    print("─" * w)
    print(f"  epsD         {out_base['epsD']:>8.4f}")
    print(f"  epsS_LR      {out_base['epsS_LR']:>8.4f}")
    print(f"  epsS_SR      {out_base['epsS_SR']:>8.4f}")
    print("─" * w)


# ========== ========== ========== ========== ==========
# 1. sweep rountines

def run_epsS_sweep(m, out_base, epsS_grid):
    """Marginal (price) elasticities over a grid of epsS_LR values."""
    w_C, w_I, r_K = [], [], []
    for epsS in epsS_grid:
        res = elas.wr_tax_elas(m, elas_out=out_base, epsS_LR=epsS)
        w_C.append(res['w_C_elas'])
        w_I.append(res['w_I_elas'])
        r_K.append(res['r_K_elas'])
    return np.array(w_C), np.array(w_I), np.array(r_K)


def run_phi_sweep_marginal(m, phi_grid, phi_restore=1.35):
    """
    Marginal (price) elasticities over a grid of phi values.
    Restores m.phi to phi_restore when done.
    """
    w_C, w_I, r_K = [], [], []
    for phi in phi_grid:
        m.phi = phi
        out = elas.dem_sup_elas(m, tau=0.0)
        res = elas.wr_tax_elas(m, elas_out=out, epsD=out['epsD'])
        w_C.append(res['w_C_elas'])
        w_I.append(res['w_I_elas'])
        r_K.append(res['r_K_elas'])
    m.phi = phi_restore
    return np.array(w_C), np.array(w_I), np.array(r_K)


def run_phi_sweep_welfare(m, phi_grid, tau_long, dlog_net_long, tauT,
                          T, tail, tau_ss=0.0, phi_restore=1.35):
    """
    Full-simulation welfare incidence over a grid of phi values.
    Restores m.phi to phi_restore when done.
    """
    w_C, w_I, r_K = [], [], []
    for phi in phi_grid:
        m.phi = phi
        sim_raw = m.solve_transition(tau_path=tau_long, tau_terminal=tauT)
        _, sim  = build_output_single.welfare_effects(
            m, sim_raw, tau_long, dlog_net_long, T=T, tail=tail, tau_ss=tau_ss
        )
        welf  = elas.welfare_incidence(sim)
        total = welf['consump_w'] + welf['investm_w'] + welf['capital_o']
        w_C.append(welf['consump_w'] / total * 100)
        w_I.append(welf['investm_w'] / total * 100)
        r_K.append(welf['capital_o'] / total * 100)
    m.phi = phi_restore
    return np.array(w_C), np.array(w_I), np.array(r_K)


def run_sweeps(m, out_base, tau_long, dlog_net_long, tauT,
               T, tail, tau_ss=0.0,
               epsS_grid=None, phi_grid=None, phi_grid_welf=None,
               phi_restore=1.35):
    """
    Run all three sweeps and return results in a single dict.

    Parameters
    ----------
    m             : calibrated CapIncModel_single
    out_base      : output of elas.dem_sup_elas(m, tau=0.0)
    tau_long      : tax path array (from shocks.perm_tc / shocks.temp_tc)
    dlog_net_long : d-log net-of-tax path
    tauT          : terminal tax rate
    T, tail       : horizon params
    tau_ss        : steady-state tau for welfare normalisation
    epsS_grid     : supply-elasticity grid  (default: linspace(0.001, 20, 2000))
    phi_grid      : phi grid for marginal sweep (default: linspace(0.001, 20, 500))
    phi_grid_welf : phi grid for welfare sweep  (default: linspace(0.001, 3, 150))
    phi_restore   : value to restore m.phi to after sweeps

    Returns
    -------
    dict with keys: epsS_grid, phi_grid, phi_grid_welf,
                    w_C_S, w_I_S, r_K_S,
                    w_C_Pm, w_I_Pm, r_K_Pm,
                    w_C_Pw, w_I_Pw, r_K_Pw,
                    epsS_base, epsD_base
    """
    if epsS_grid    is None: epsS_grid    = np.linspace(0.001, 20,  2000)
    if phi_grid     is None: phi_grid     = np.linspace(0.001, 20,   500)
    if phi_grid_welf is None: phi_grid_welf = np.linspace(0.001,  3,  150)

    print("Running epsS sweep …")
    w_C_S, w_I_S, r_K_S = run_epsS_sweep(m, out_base, epsS_grid)

    print("Running phi marginal sweep …")
    w_C_Pm, w_I_Pm, r_K_Pm = run_phi_sweep_marginal(m, phi_grid, phi_restore)

    print("Running phi welfare sweep …")
    w_C_Pw, w_I_Pw, r_K_Pw = run_phi_sweep_welfare(
        m, phi_grid_welf, tau_long, dlog_net_long, tauT,
        T, tail, tau_ss, phi_restore
    )
    print("Done.")

    return dict(
        epsS_grid=epsS_grid,
        phi_grid=phi_grid,
        phi_grid_welf=phi_grid_welf,
        w_C_S=w_C_S,   w_I_S=w_I_S,   r_K_S=r_K_S,
        w_C_Pm=w_C_Pm, w_I_Pm=w_I_Pm, r_K_Pm=r_K_Pm,
        w_C_Pw=w_C_Pw, w_I_Pw=w_I_Pw, r_K_Pw=r_K_Pw,
        epsS_base=out_base['epsS_LR'],
        epsD_base=out_base['epsD'],
    )


# ========== ========== ========== ========== ==========
# 2. plotting routine

def plot_sweep(sweep_res, savepath='0_output/main_arg.png'):
    """
    Reproduce the two-panel sweep figure.

    Parameters
    ----------
    sweep_res : dict returned by run_sweeps()
    savepath  : output path
    """
    colors       = ['crimson', 'k', '#00B8D9']
    colors_light = [lighten_color(c, 0.4) for c in colors]

    epsS_grid     = sweep_res['epsS_grid']
    phi_grid      = sweep_res['phi_grid']
    phi_grid_welf = sweep_res['phi_grid_welf']

    w_C_S,  w_I_S,  r_K_S  = sweep_res['w_C_S'],  sweep_res['w_I_S'],  sweep_res['r_K_S']
    w_C_Pm, w_I_Pm, r_K_Pm = sweep_res['w_C_Pm'], sweep_res['w_I_Pm'], sweep_res['r_K_Pm']
    w_C_Pw, w_I_Pw, r_K_Pw = sweep_res['w_C_Pw'], sweep_res['w_I_Pw'], sweep_res['r_K_Pw']

    # share versions (already %-scaled for welfare; recompute for marginal)
    total_S  = w_C_S  + w_I_S  + r_K_S
    total_Pm = w_C_Pm + w_I_Pm + r_K_Pm

    legend_handles = [
        Patch(color=colors[0], alpha=0.9, label=r'Capitalists ($r_K$)'),
        Line2D([0], [0], color='gray', lw=2, ls='-',  label=r'Varying $\varepsilon^S$'),
        Patch(color=colors[2], alpha=0.9, label=r'Investment workers ($w_I$)'),
        Line2D([0], [0], color='gray', lw=2, ls='--', alpha=0.5, label=r'Varying $\phi$'),
        Patch(color=colors[1], alpha=0.9, label=r'Consumption workers ($w_C$)'),
        Patch(color='gray', alpha=0.15, label=r'Calibration area'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # ========== ========== ========== ========== ==========
    # Left: marginal tax elasticities sweep 
    ax = axes[0]

    phi_mapped = np.linspace(epsS_grid[0], epsS_grid[-1], len(phi_grid))
    ax.plot(phi_mapped, w_C_Pm, color=colors_light[1], lw=2, ls='--', alpha=0.7)
    ax.plot(phi_mapped, w_I_Pm, color=colors_light[2], lw=2, ls='--', alpha=0.7)
    ax.plot(phi_mapped, r_K_Pm, color=colors_light[0], lw=2, ls='--', alpha=0.7)

    ax.plot(epsS_grid, w_C_S, color=colors[1], lw=2, ls='-')
    ax.plot(epsS_grid, w_I_S, color=colors[2], lw=2, ls='-')
    ax.plot(epsS_grid, r_K_S, color=colors[0], lw=2, ls='-')

    # ax.axvline(0.65, color="#4A4A5A", lw=2, ls='-.')
    ax.axvspan(0.5, 1, color='gray', alpha=0.15, zorder=0)

    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels([r'$\phi=0$', r'$5$', r'$10$', r'$15$', r'$20$'])
    ax.set_ylabel('Tax elasticity')
    ax.set_title(r'Tax elasticity — varying $\varepsilon^S$ or $\phi$', pad=10)
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(0, 20)
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_xticklabels([r'$\varepsilon^S=0$', r'$5$', r'$10$', r'$15$', r'$20$'])

    # ========== ========== ========== ========== ==========
    # Right: welfare incidence sweep
    ax = axes[1]

    ax.stackplot(phi_grid_welf, w_C_Pw, w_I_Pw, r_K_Pw,
                 colors=[colors[1], colors[2], colors[0]], alpha=0.8)
    # ax.axvline(0.65, color="#4A4A5A", lw=2, ls='-.')
    ax.axvspan(0.5, 1, color='gray', alpha=0.45, zorder=0)

    ax.set_ylabel('Share of total (%)')
    ax.set_title(r'Welfare incidence — varying $\phi$', pad=35)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.6)

    ax.set_xticks([0, 1.0, 2.0, 3.0])
    ax.set_xticklabels([r'$\phi=0$', r'$1$', r'$2$', r'$3$'])

    # ========== ========== ========== ========== ==========
    # legend
    fig.legend(handles=legend_handles,
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               frameon=True, ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    plt.savefig(savepath, dpi=200)
    plt.show()