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

        # 4.2 after-tax capital income at SS
        D_ss = (1 - tau_ss) * st_ss["rC_gross"] * ss["K"] 

        # 4.3 compute welfare gains
        wg_C = (WB_C_1 * np.log(sim["w1C"] / st_ss["w1C"]) +
                WB_C_2 * np.log(sim["w2C"] / st_ss["w2C"]))
        wg_I = (WB_I_1 * np.log(sim["w1I"] / st_ss["w1I"]) +
                WB_I_2 * np.log(sim["w2I"] / st_ss["w2I"]))
        wg_K = D_ss * dlog_net # old

        # 4.4 alternative wg_K, here as last claimant
        # 4.4.1 make sure these are (T+1,)
        # Y_ss = ss["C"] + ss["pI"] * ss["I"]
        # WB_ss = WB_C_1 + WB_C_2 + WB_I_1 + WB_I_2
        # D_ss  = (1 - tau_ss) * (Y_ss - WB_ss)

        # wg_L_path = np.asarray(wg_C, float) + np.asarray(wg_I, float) 
        # WG_total_path = D_ss * dlog_net 

        # # 4.4.2. residual claimant
        # wg_K = WG_total_path - wg_L_path          



        # 4.5 ... in ss consumption units
        C_ss = ss["C"]
        wC_pct = 100 * wg_C / C_ss
        wI_pct = 100 * wg_I / C_ss
        wK_pct = 100 * wg_K / C_ss 

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