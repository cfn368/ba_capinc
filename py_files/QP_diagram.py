"""
Capital market equilibrium — quantitative model
Two panels: (a) NGM  (b) Baseline calibration with varying phi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from py_files.elas import dem_sup_elas
from py_files.capinc_single import CapIncModel_single


# ── colours ──────────────────────────────────────────────────────────────────
C_SUPPLY  = "#1a1a2e"   # dark navy  — supply curve
C_DEMAND  = "#c0392b"   # crimson    — demand curve
C_NEW     = "k"   # green      — new equilibrium
C_OLD     = "k"   # grey       — old equilibrium


# ── baseline calibration ──────────────────────────────────────────────────────
GENERAL = dict(
    alphaK=0.49, alphaL=0.51,
    betaK=0.35,  betaL=0.65,
    phi=0.75,
)
# GENERAL = dict(
#     betaK=0.49, betaL=0.51,
#     alphaK=0.35,  alphaL=0.65,
#     phi=0.75,
# )


DTAU = 0.10   # illustrative 10 pp tax cut


# ── model factory ─────────────────────────────────────────────────────────────
def make_model(calib, r=0.07, delta=0.15, theta=0.25, mu=0.26):
    m = CapIncModel_single()
    m.r = r; m.delta = delta; m.theta = theta; m.mu = mu
    for k, v in calib.items():
        setattr(m, k, v)
    return m


# ── elasticities + equilibrium given a model + dtau ──────────────────────────
def get_eq(m, dtau, ngm=False):
    elas  = dem_sup_elas(m, tau=0.0)
    eps_D = float(elas["epsD"])
    eps_S = np.inf if ngm else float(elas["epsS_LR"])

    log_K_shift = eps_D / (1.0 - m.alphaK) * dtau

    if ngm:
        K_new = np.exp(log_K_shift); q_new = 1.0
    else:
        log_K_new = (log_K_shift / eps_D) / (1/eps_S + 1/eps_D)
        K_new = np.exp(log_K_new)
        q_new = np.exp(log_K_new / eps_S)

    return eps_S, eps_D, log_K_shift, K_new, q_new


# ── curve helpers ─────────────────────────────────────────────────────────────
def supply_curve(K_grid, eps_S):
    if np.isinf(eps_S):
        return np.ones_like(K_grid)
    return K_grid ** (1.0 / eps_S)

def demand_curve(K_grid, eps_D, log_K_shift=0.0):
    return (K_grid * np.exp(-log_K_shift)) ** (-1.0 / eps_D)


# ── panel (a): NGM ────────────────────────────────────────────────────────────
def draw_ngm(ax):
    m = make_model(GENERAL)
    eps_S, eps_D, log_K_shift, K_new, q_new = get_eq(m, DTAU, ngm=True)

    K_grid = np.linspace(0.5, 2.0, 500)

    ax.plot(K_grid, supply_curve(K_grid, np.inf),            color=C_SUPPLY, lw=2, zorder=3)
    ax.plot(K_grid, demand_curve(K_grid, 1.5),               color=C_DEMAND, lw=2, zorder=3)
    ax.plot(K_grid, demand_curve(K_grid, 1.5, log_K_shift),  color=C_DEMAND, lw=2, ls=":", zorder=3)

    _mark_eq(ax, 1.0, 1.0, K_new, q_new)

    ax.text(0.97, 0.05,
            "$\\varepsilon^S = +\\infty$\n$\\varepsilon^D = {:.2f}$".format(1.5),
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbb"))

    _format_ax(ax, "Neoclassical case: $\\alpha_K = \\beta_K$", ylabel=True)
    ax.title.set_fontsize(15)


# ── panel (b): baseline calibration ──────────────────────────────────────────
def draw_baseline(ax):
    m = make_model(GENERAL)
    eps_S, eps_D, log_K_shift, K_new, q_new = get_eq(m, DTAU)

    K_grid = np.linspace(0.5, 2.0, 500)

    ax.plot(K_grid, supply_curve(K_grid, eps_S),               color=C_SUPPLY, lw=2, zorder=3)
    ax.plot(K_grid, demand_curve(K_grid, eps_D),               color=C_DEMAND, lw=2, zorder=3)
    ax.plot(K_grid, demand_curve(K_grid, eps_D, log_K_shift),  color=C_DEMAND, lw=2, ls=":", zorder=3)

    _mark_eq(ax, 1.0, 1.0, K_new, q_new)

    ax.text(0.97, 0.05,
            "$\\varepsilon^S = 1.46$\n$\\varepsilon^D = 1.34$".format(eps_S, eps_D),
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#bbb"))

    _format_ax(ax, "Intangible era $\\alpha_L < \\beta_L$", ylabel=False)
    ax.title.set_fontsize(15)


# ── shared helpers ────────────────────────────────────────────────────────────
def _mark_eq(ax, K_old, q_old, K_new, q_new):
    for K_eq, q_eq, col in [(K_old, q_old, C_OLD), (K_new, q_new, C_NEW)]:
        ax.scatter([K_eq], [q_eq], color=col, s=48, zorder=5, marker='s')
        ax.axvline(K_eq, color=col, lw=1, ls="--", alpha=0.55)
        ax.axhline(q_eq, color=col, lw=1, ls="--", alpha=0.55)

def _format_ax(ax, title, ylabel=True):
    ax.set_title(title, pad=7)
    ax.set_xlabel(r"Quantity $K$")
    if ylabel:
        ax.set_ylabel(r"Relative price $q$")
    ax.set_xlim(0.55, 1.95)
    ax.set_ylim(0.60, 1.70)
    ax.grid(True, alpha=0.22, ls="--")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

# ── shared bottom legend ──────────────────────────────────────────────────────
def add_legend(fig):
    handles = [
        Line2D([0], [0], color=C_SUPPLY, lw=2, label="Capital Supply"),
        Line2D([0], [0], color=C_DEMAND, lw=2, label="Capital Demand"),
        Line2D([0], [0], color=C_DEMAND, lw=2, ls=":", label="Tax cut"),
        Line2D([0], [0], color=C_OLD, lw=0, marker="s", ms=6, label="Eq."),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))


# ── main entry point ──────────────────────────────────────────────────────────
def make_figure(save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.06, bottom=0.18)

    draw_ngm(axes[0])
    draw_baseline(axes[1])
    add_legend(fig)

    if save_path:
        import pathlib
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}{{.png}}")
    return fig


if __name__ == "__main__":
    make_figure(save_path="output/cap_market_quant")