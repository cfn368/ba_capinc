import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import py_files.var_groups as var_groups


IRF_COLOR = "#4A4A5A"
COLOR_C   = "k"
COLOR_I   = "#00B8D9"
COLOR_K   = "crimson"


def plot_irf(sim1, sim2, sim3, sim_welfare, T_plot, C_ss, savepath="0_output/sim_lr_cs.png"):
    """
    Plot IRF panels for standard model variables and welfare gains.

    Parameters
    ----------
    sim1        : dict  – IRF simulation (permanent shock path)
    sim2        : dict  – baseline SS simulation
    sim3        : dict  – new SS simulation
    sim_welfare : dict  – welfare output from build_output_single.welfare_effects
                          (must contain wg_C, wg_I, wg_K)
    T_plot      : int   – number of periods to plot
    C_ss        : float – baseline SS consumption level (for welfare normalisation)
    savepath    : str   – file path for saved figure
    """
    welfare_colors = {"wg_C": COLOR_C, "wg_I": COLOR_I, "wg_K": COLOR_K}

    h = np.arange(T_plot)

    def trunc_pack(sim):
        return {k: np.asarray(v) for k, v in sim.items()}

    S1 = trunc_pack({**sim1, **sim_welfare})
    S2 = trunc_pack(sim2)
    S3 = trunc_pack(sim3)

    keys = ["tau", "q", "pI", "K", "sK", "sL", "wC", "wI"]
    keys = [k for k in keys if k in S1 and k in S2]

    welfare_keys = [
        ("wg_I", r"$\text{WG}^L_I$"),
        ("wg_C", r"$\text{WG}^L_C$"),
        ("wg_K", r"$\text{WG}^K$"),
    ]

    log_dev_keys = {"pI", "q", "K", "I", "C", "wC", "wI"}

    def to_log_dev(series, baseline_series):
        ss_val = float(baseline_series[0])
        return 100.0 * np.log(np.asarray(series, float) / ss_val)

    def to_level_dev(series, baseline_series):
        ss_val = float(baseline_series[0])
        return np.asarray(series, float) - ss_val

    ncols = 4
    nrows = 3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 2.6 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    # legend handles
    h1   = plt.Line2D([0], [0], color=IRF_COLOR, lw=3, ls="-.")
    h2   = plt.Line2D([0], [0], color="k",        lw=1.5)
    h3   = plt.Line2D([0], [0], color="k",        lw=1.5, ls=":", alpha=0.7)
    h4_K = plt.Line2D([0], [0], color=COLOR_K,    lw=3,   ls="-")
    h4_I = plt.Line2D([0], [0], color=COLOR_I,    lw=3,   ls="-")
    h4_C = plt.Line2D([0], [0], color=COLOR_C,    lw=3,   ls="-")

    # turn off unused axes
    for j in range(len(keys) + len(welfare_keys), len(axes)):
        axes[j].axis("off")

    # standard variable panels
    for i, k in enumerate(keys):
        ax = axes[i]
        raw1 = S1[k][:T_plot]
        raw2 = S2[k][:T_plot]
        raw3 = S3[k][:T_plot]

        if k in log_dev_keys:
            y1, y2, y3 = to_log_dev(raw1, raw2), to_log_dev(raw2, raw2), to_log_dev(raw3, raw2)
            ylabel = "log dev. (%)"
        else:
            y1, y2, y3 = to_level_dev(raw1, raw2), to_level_dev(raw2, raw2), to_level_dev(raw3, raw2)
            ylabel = "level dev."

        ax.plot(h, y1, lw=3, color=IRF_COLOR, ls="-.")
        ax.plot(h, y2, lw=2, color="k")
        ax.plot(h, y3, lw=2, color="k", ls=":")
        ax.axhline(0, color="k", lw=1, ls="-")
        ax.set_title(var_groups.model_var.get(k, k))
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune=None))
        ax.tick_params(axis="y", labelsize=10)
        ticks = ax.get_yticks()
        ticks[np.argmin(np.abs(ticks))] = 0.0
        ax.set_yticks(ticks)

    # welfare panels
    for j, (wk, title) in enumerate(welfare_keys):
        ax = axes[len(keys) + j]
        path = 100.0 * np.asarray(S1[wk], float)[:T_plot] / C_ss
        ax.plot(h, path, lw=3, color=welfare_colors[wk])
        ax.axhline(0, color="k", lw=2, ls="-")
        ax.set_title(title)
        ax.set_ylabel("% of SS cons.")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ticks = ax.get_yticks()
        ticks[np.argmin(np.abs(ticks))] = 0.0
        ax.set_yticks(ticks)

    fig.legend(
        [h1, h2, h3, h4_I, h4_C, h4_K],
        ["IRF", "old ss", "new ss", r"Investment w.", r"Consumption w.", "Capitalists"],
        loc="lower center", ncol=6, frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )

    for col in range(ncols):
        col_axes = [axes[row * ncols + col] for row in range(nrows)
                    if row * ncols + col < len(axes)]
        active = [ax for ax in col_axes if ax.get_visible() and ax.axison]
        if active:
            active[-1].tick_params(axis="x", labelbottom=True)

    plt.savefig(savepath, dpi=200)
    plt.show()
