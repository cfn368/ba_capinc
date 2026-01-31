import numpy as np
import matplotlib.pyplot as plt

###########################################################
# 1. paper like wealth effects
###########################################################

def welfare_effects(m, sim_raw, tau_long, dlog_net_long, 
                    T=25, tail=50, tau_ss=None,
                    ):

        # 1. plot only slice of full simulation timeline
        ss = m.solve_steady_state(tau=tau_ss)

        sl = slice(0, T + 1)
        T_solve = int(T + tail)
        sim = {k: (np.asarray(v)[sl] if isinstance(v, (list, np.ndarray)) and len(v) == T_solve + 1 else v)
                for k, v in sim_raw.items()}
        
        # 2. get relevant
        dlog_net = dlog_net_long[sl]
        h = np.arange(T + 1)

        # 3. left panel: percent deviations
        pct = lambda x, xss: 100 * np.log(np.asarray(x) / float(xss))

        dq  = pct(sim["q"],  ss["q"])   # relative price of capital good
        dpI = pct(sim["pI"], ss["pI"])  # investment price
        dK  = pct(sim["K"],  ss["K"])   # capital stock

        # 4. right panel: undiscounted welfare effects
        # ... normalised by C_ss
        m.z_last[:] = 0.0
        st_ss = m._static(ss["K"], ss["q"], tau=tau_ss)

        # 4.1 wage bill
        WB_C_1 = st_ss["w1C"] * st_ss["L1C"]
        WB_C_2 = st_ss["w2C"] * st_ss["L2C"]
        WB_I_1 = st_ss["w1I"] * st_ss["L1I"]
        WB_I_2 = st_ss["w2I"] * st_ss["L2I"]

        # 4.3 compute welfare gains (paper intention):
        # d log wages vs baseline SS
        dlog_w1C = np.log(np.asarray(sim["w1C"], float) / float(st_ss["w1C"]))
        dlog_w2C = np.log(np.asarray(sim["w2C"], float) / float(st_ss["w2C"]))
        dlog_w1I = np.log(np.asarray(sim["w1I"], float) / float(st_ss["w1I"]))
        dlog_w2I = np.log(np.asarray(sim["w2I"], float) / float(st_ss["w2I"]))

        # sector worker welfare-flow paths
        wg_C = (np.asarray(sim["w1C"], float) * dlog_w1C * np.asarray(sim["L1C"], float) +
                np.asarray(sim["w2C"], float) * dlog_w2C * np.asarray(sim["L2C"], float))

        wg_I = (np.asarray(sim["w1I"], float) * dlog_w1I * np.asarray(sim["L1I"], float) +
                np.asarray(sim["w2I"], float) * dlog_w2I * np.asarray(sim["L2I"], float))

        # 4.4 alternative wg_K, here as last claimant
        # time-t value added in consumption units
        Y_t = np.asarray(sim["C"], float) + np.asarray(sim["pI"], float) * np.asarray(sim["I"], float)

        # time-t wage bill (use time-t wages and time-t labor)
        WB_t = (np.asarray(sim["w1C"], float) * np.asarray(sim["L1C"], float) +
                np.asarray(sim["w2C"], float) * np.asarray(sim["L2C"], float) +
                np.asarray(sim["w1I"], float) * np.asarray(sim["L1I"], float) +
                np.asarray(sim["w2I"], float) * np.asarray(sim["L2I"], float))

        # time-t dividends, paper definition
        tau_t = np.asarray(tau_long, float)[sl]          # ensure aligned (T+1,)
        D_t = (1.0 - tau_t) * (Y_t - WB_t)               # (T+1,)

        # workers total welfare-flow path
        wg_L_path = np.asarray(wg_C, float) + np.asarray(wg_I, float)

        # total welfare-flow path (paper step: D_t * dlog(1-τ_t))
        WG_total_path = D_t * np.asarray(dlog_net, float)

        # residual claimant
        wg_K = WG_total_path - wg_L_path    

        # 4.5 ... in ss consumption units
        wC_pct = 100 * wg_C / ss["C"]
        wI_pct = 100 * wg_I / ss["C"]
        wK_pct = 100 * wg_K / ss["C"]

        # 4.6 append solution dict
        sim["dq_pct"]  = np.asarray(dq, float)
        sim["dpI_pct"] = np.asarray(dpI, float)
        sim["dK_pct"]  = np.asarray(dK, float)

        sim["wC_pct"]  = np.asarray(wC_pct, float)
        sim["wI_pct"]  = np.asarray(wI_pct, float)
        sim["wK_pct"]  = np.asarray(wK_pct, float)

        # 5. plot
        fig, (ax1, ax2) = plt.subplots(1, 2, 
                                        figsize=(12, 6), 
                                        constrained_layout=True,
        )

        # (a) Capital and valuations
        ax1.plot(h, dq, color='k', ls="-",  lw=2, label="value of installed capital $q$")
        ax1.plot(h, dpI, color='k', ls=":",  lw=2, label="price of capital good $p_I$")
        ax1.plot(h, dK,  "-",   lw=2, label="capital stock $K$", color='crimson')
        ax1.axhline(0, color="k", ls=":", lw=1, alpha=0.6)
        ax1.set_xlabel("horizon")
        ax1.set_ylabel("deviation (%)")
        ax1.set_title("(a) Capital and valuations")
        ax1.legend(frameon=True)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        # (b) Welfare effects
        ax2.plot(h, wC_pct, lw=2, label="consumption sector workers", color="k")
        ax2.plot(h, wI_pct, lw=2, label="investment sector workers",  color='#00B8D9')
        ax2.plot(h, wK_pct, lw=2, label="capitalists",                color='crimson')
        ax2.axhline(0, color="k", ls=":", lw=1, alpha=0.6)
        ax2.set_xlabel("horizon")
        ax2.set_ylabel("welfare gain / consumption (%)")
        ax2.set_title("(b) Welfare effects")
        ax2.legend(frameon=True)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

        return fig, (ax1, ax2), ss, sim

###########################################################
# 2. labour income share
###########################################################

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