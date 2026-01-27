import numpy as np
import matplotlib.pyplot as plt

def net_tax_path_power(T=25, tau0=0.32, tauT=0.22, kappa=0.4, T_reform=25):
    T = int(T)
    T_reform = int(T_reform)
    if T_reform < 1 or T_reform > T:
        raise ValueError("Need 1 <= T_reform <= T.")

    t = np.arange(T + 1)

    # progress: 0->1 over first T_reform, then flat at 1
    s = np.empty(T + 1, dtype=float)
    s[:T_reform + 1] = (t[:T_reform + 1] / T_reform) ** kappa
    s[T_reform + 1:] = 1.0

    net0 = 1.0 - tau0
    netT = 1.0 - tauT

    log_net = (1.0 - s) * np.log(net0) + s * np.log(netT)
    net_t = np.exp(log_net)
    tau_t = 1.0 - net_t
    dlog  = np.log(net_t / net0)
    return net_t, tau_t, dlog


def plot_figure5_reform(m, T=25, tau_ss=0.32, tauT=0.22, kappa=0.4, ref=False):
    # 1) steady state at initial tax (baseline you deviate from)
    ss = m.solve_steady_state(tau=tau_ss)

    # 2) smooth reform path + transition
    net_t, tau_t, dlog_net = net_tax_path_power(T=T, tau0=tau_ss, tauT=tauT, kappa=kappa, T_reform=25)

    # terminal closure: use steady state q at tauT (via tau_terminal)
    sim = m.solve_transition(tau_path=tau_t, tau_terminal=tauT)

    h = np.arange(T + 1)

    # 3) percent deviations (panel a) vs initial SS
    pct = lambda x, xss: 100 * np.log(np.asarray(x) / float(xss))
    dq  = pct(sim["q"],  ss["q"])
    dpI = pct(sim["pI"], ss["pI"])
    dK  = pct(sim["K"],  ss["K"])

    # 4) “undiscounted welfare effects” (panel b), normalized by C_ss
    # weights are from initial SS (envelope-style)
    m.z_last[:] = 0.0
    st_ss = m._static(ss["K"], ss["q"], tau=tau_ss)

    WB_C_1 = st_ss["w1C"] * st_ss["L1C"]
    WB_C_2 = st_ss["w2C"] * st_ss["L2C"]
    WB_I_1 = st_ss["w1I"] * st_ss["L1I"]
    WB_I_2 = st_ss["w2I"] * st_ss["L2I"]

    D_ss = (1 - tau_ss) * st_ss["rC_gross"] * ss["K"]

    wg_C = (WB_C_1 * np.log(sim["w1C"] / st_ss["w1C"]) +
            WB_C_2 * np.log(sim["w2C"] / st_ss["w2C"]))
    wg_I = (WB_I_1 * np.log(sim["w1I"] / st_ss["w1I"]) +
            WB_I_2 * np.log(sim["w2I"] / st_ss["w2I"]))
    wg_K = D_ss * dlog_net

    C_ss = ss["C"]
    wC_pct = 100 * wg_C / C_ss
    wI_pct = 100 * wg_I / C_ss
    wK_pct = 100 * wg_K / C_ss

    # 5) plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   constrained_layout=True)

    ax1.plot(h, dq,  color='k', ls="-", lw=2, label="value of installed capital $q$")
    ax1.plot(h, dpI, color='k', ls=":", lw=2, label="price of capital good $p_I$")
    ax1.plot(h, dK,  "-", lw=2, label="capital stock $K$", color='crimson')
    ax1.axhline(0, color="k", ls=":", lw=1, alpha=0.6)
    ax1.set_xlabel("horizon")
    ax1.set_ylabel("deviation (%)")
    ax1.set_title(r"(a) Capital and valuations (32% $\rightarrow$ 22%)")
    ax1.legend(frameon=True)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2.plot(h, wC_pct, lw=2, label="consumption sector workers", color="k")
    ax2.plot(h, wI_pct, lw=2, label="investment sector workers", color='#00B8D9')
    ax2.plot(h, wK_pct, lw=2, label="capitalists", color='crimson')
    ax2.axhline(0, color="k", ls=":", lw=1, alpha=0.6)
    ax2.set_xlabel("horizon")
    ax2.set_ylabel("welfare gain / consumption (%)")
    ax2.set_title(r"(b) Welfare effects (vs SS at 32%)")
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    return fig, (ax1, ax2), ss, sim, {"tau_t": tau_t, "net_t": net_t, "dlog_net": dlog_net}

def labour_share(m, sim, gamma=1):
    # 1) value added in consumption units
    C  = np.asarray(sim["C"], float)
    I  = np.asarray(sim["I"], float)
    pI = np.asarray(sim["pI"], float)
    Y  = C + pI * I

    # 2) value weights of sectors
    wI = (pI * I) / Y
    wC = C / Y

    LS_C = (1 - m.alphaK) * wC
    LS_I = (1 - m.betaK) * wI

    # 3) aggregate labor share (competitive Cobb–Douglas within each sector)
    LS = LS_C + LS_I
    LS_gamma = LS_C + gamma*LS_I
    return {"Y": Y, "wC": wC, "wI": wI, 
            "LS": LS, 'LS_C': LS_C, 'LS_I': LS_I,
            'LS_gamma': LS_gamma, 'pII': pI * I,
            'C': C,
        }