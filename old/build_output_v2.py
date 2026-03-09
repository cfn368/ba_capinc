import numpy as np
import matplotlib.pyplot as plt

###########################################################
# 1. paper like wealth effects
###########################################################

def welfare_effects(m, sim_raw, tau_long, dlog_net_long,
                    T=25, tail=50, tau_ss=None):

        # 0) horizons
        T = int(T); tail = int(tail)
        T_solve = int(T + tail)
        sl_full = slice(0, T_solve + 1)
        sl_plot = slice(0, T + 1)

        # 1) steady state
        ss = m.solve_steady_state(tau=float(tau_ss))

        # 2) coerce sim to full horizon arrays (NO truncation yet)
        sim_full = {
                k: (np.asarray(v)[sl_full] if isinstance(v, (list, np.ndarray)) and len(v) >= T_solve + 1 else v)
                for k, v in sim_raw.items()
        }

        # 3) plot horizon index
        h = np.arange(T + 1)

        # 4) percent deviations for plotting (computed from full then sliced)
        pct = lambda x, xss: 100 * np.log(np.asarray(x, float) / float(xss))
        dq_full  = pct(sim_full["q"],  ss["q"])
        dpI_full = pct(sim_full["pI"], ss["pI"])
        dK_full  = pct(sim_full["K"],  ss["K"])

        dq  = dq_full[sl_plot]
        dpI = dpI_full[sl_plot]
        dK  = dK_full[sl_plot]

        # 5) SS static (baseline objects)
        m.z_last[:] = 0.0
        st_ss = m._static(ss["K"], ss["q"], tau=float(tau_ss))

        # SS wage bills and labor (envelope theorem: hold L at baseline optimum)
        w1C_ss, L1C_ss = float(st_ss["w1C"]), float(st_ss["L1C"])
        w2C_ss, L2C_ss = float(st_ss["w2C"]), float(st_ss["L2C"])
        w1I_ss, L1I_ss = float(st_ss["w1I"]), float(st_ss["L1I"])
        w2I_ss, L2I_ss = float(st_ss["w2I"]), float(st_ss["L2I"])

        # 6) FULL-HORIZON welfare paths
        # d log wages vs baseline SS (full horizon)
        dlog_w1C_full = np.log(np.asarray(sim_full["w1C"], float) / w1C_ss)
        dlog_w2C_full = np.log(np.asarray(sim_full["w2C"], float) / w2C_ss)
        dlog_w1I_full = np.log(np.asarray(sim_full["w1I"], float) / w1I_ss)
        dlog_w2I_full = np.log(np.asarray(sim_full["w2I"], float) / w2I_ss)

        # worker welfare-flow paths (envelope-consistent, SS weights)
        wg_C_full = (w1C_ss * L1C_ss) * dlog_w1C_full + (w2C_ss * L2C_ss) * dlog_w2C_full
        wg_I_full = (w1I_ss * L1I_ss) * dlog_w1I_full + (w2I_ss * L2I_ss) * dlog_w2I_full
        wg_L_full = wg_C_full + wg_I_full

        # net-of-tax log change (FULL). You can ignore dlog_net_long input here to avoid mismatch bugs.
        tau_t_full = np.asarray(tau_long, float)[sl_full]
        dlog_net_full = np.log((1.0 - tau_t_full) / (1.0 - float(tau_ss)))

        # constant SS dividends D (scalar)
        C_ss  = float(st_ss["C"])
        pI_ss = float(st_ss["pI"])
        I_ss  = float(st_ss["I"])
        Y_ss  = C_ss + pI_ss * I_ss
        WB_ss = (w1C_ss * L1C_ss + w2C_ss * L2C_ss + w1I_ss * L1I_ss + w2I_ss * L2I_ss)
        D_ss  = (1.0 - float(tau_ss)) * (Y_ss - WB_ss)   # scalar

        WG_total_full = D_ss * dlog_net_full 
        wg_K_full = WG_total_full - wg_L_full

        # 7) Table values: use FULL horizon sums (or PV if you later add discounting)
        
        # add discounting
        t_full = np.arange(T_solve + 1)
        beta = 1/(1 + m.r) ** t_full

        pv = lambda x: float(np.sum(beta * np.asarray(x, float)))

        C_norm = float(ss["C"])

        table = {
        "pv_wg_C": pv(wg_C_full),
        "pv_wg_I": pv(wg_I_full),
        "pv_wg_K": pv(wg_K_full),
        "pv_WG":   pv(WG_total_full),

        "wg_C_total_pctC": 100.0 * pv(wg_C_full) / C_norm,
        "wg_I_total_pctC": 100.0 * pv(wg_I_full) / C_norm,
        "wg_K_total_pctC": 100.0 * pv(wg_K_full) / C_norm,
        "WG_total_pctC":   100.0 * pv(WG_total_full) / C_norm,
        }

        # 8) Plot arrays: slice AFTER computing full welfare
        wC_pct = 100.0 * wg_C_full[sl_plot] / C_norm
        wI_pct = 100.0 * wg_I_full[sl_plot] / C_norm
        wK_pct = 100.0 * wg_K_full[sl_plot] / C_norm
        WG_pct = 100.0 * WG_total_full[sl_plot] / C_norm

        # 9) build sim dict for return (plot horizon only, but include full-table scalars)
        sim = {k: (v[sl_plot] if isinstance(v, np.ndarray) and v.shape[0] == T_solve + 1 else v)
                for k, v in sim_full.items()}

        sim["dq_pct"]  = dq
        sim["dpI_pct"] = dpI
        sim["dK_pct"]  = dK

        sim["wg_C"] = wg_C_full[sl_plot]
        sim["wg_I"] = wg_I_full[sl_plot] 
        sim["wg_K"] = wg_K_full[sl_plot]
        sim["WG_total_path"] = WG_total_full[sl_plot]

        sim["table"] = table  

        # 10) plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        ax1.plot(h, dq,  color='k', ls="-", lw=2, label="value of installed capital $q$")
        ax1.plot(h, dpI, color='k', ls=":", lw=2, label="price of capital good $p_I$")
        ax1.plot(h, dK,  color='crimson', ls="-", lw=2, label="capital stock $K$")
        ax1.axhline(0, color="k", ls=":", lw=1, alpha=0.6)
        ax1.set_xlabel("horizon")
        ax1.set_ylabel("deviation (%)")
        ax1.set_title("(a) Capital and valuations")
        ax1.legend(frameon=True)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        ax2.plot(h, wC_pct, lw=2, label="consumption sector workers", color="k")
        ax2.plot(h, wI_pct, lw=2, label="investment sector workers",  color="#00B8D9")
        ax2.plot(h, wK_pct, lw=2, label="capitalists",                color="crimson")
        ax2.plot(h, WG_pct, lw=2, label="Total wealth gain",          color="gray", ls=":")
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

###########################################################
# 3. incidence and elasticities
###########################################################

def inc_elas(m, sim, tau):

        # ========== ========== ========== ========== ========== 
        # 1. tax incidence 
        consump_w = sim["table"]["pv_wg_C"] / sim["table"]["pv_WG"]
        investm_w = sim["table"]["pv_wg_I"] / sim["table"]["pv_WG"]
        capital_o = sim["table"]["pv_wg_K"] / sim["table"]["pv_WG"]

        # consump_w = sim['wg_C'].sum() / sim['WG_total_path'].sum()
        # investm_w = sim['wg_I'].sum() / sim['WG_total_path'].sum()
        # capital_o = sim['wg_K'].sum() / sim['WG_total_path'].sum()

        print("\n" + "="*44)
        print("-"*44)
        print(" Incidence (share of total welfare gain) ")
        print("-"*44)
        print(f"{'Consumption workers':<22} {consump_w:>6.1%}")
        print(f"{'Investment workers':<22} {investm_w:>6.1%}")
        print(f"{'Capitalists':<22} {capital_o:>6.1%}")

        # ========== ========== ========== ========== ========== 
        # 2. elasticities (q,K) 
        ss = m.solve_steady_state(tau=tau)
        vals = m.static_block_sigmoid(K=ss['K'], q=ss['q'], tau=ss['tau'])

        A_theta = np.array([
        [ 
                vals['sK']/(1-vals['sK']) * (m.alphaK-1) + (m.betaK-1)  ,
                vals['s1']/(1-vals['s1']) * m.alpha1L + m.beta1L        ,
                vals['s2']/(1-vals['s2']) * m.alpha2L + m.beta2L        ,
                1.0
        ]   ,
        [
                vals['sK']/(1-vals['sK']) * m.alphaK + m.betaK          ,
                vals['s1']/(1-vals['s1']) * (m.alpha1L-(1+m.phi1)/m.phi1)
                + m.beta1L - (1+m.phi1)/m.phi1                          ,
                vals['s2']/(1-vals['s2']) * m.alpha2L + m.beta2L        ,
                1.0
        ]   ,
        [
                vals['sK']/(1-vals['sK']) * m.alphaK + m.betaK          ,
                vals['s1']/(1-vals['s1']) * m.alpha1L + m.beta1L        ,
                vals['s2']/(1-vals['s2']) * (m.alpha2L-(1+m.phi2)/m.phi2)
                + m.beta2L - (1+m.phi2)/m.phi2                          ,
                1.0
        ]   ,
        [
                m.theta*m.betaK, m.theta*m.beta1L, m.theta*m.beta2L, 1.0
        ]
        ])

        A0 = A_theta.copy()
        A0[3, :] = np.array([0.0, 0.0, 0.0, 1.0])

        b = np.array([0.0, 0.0, 0.0, 1.0])

        x_theta = np.linalg.solve(A_theta, b)
        x0      = np.linalg.solve(A0, b)

        supply_vec = np.array([ m.betaK, m.beta1L, m.beta2L, 0.0 ])
        demand_vec = np.array([
        vals['sK']/(1-vals['sK'])                                   , 
        -m.alpha1L/(1-m.alphaK) * vals['s1']/(1-vals['s1'])         ,
        -m.alpha2L/(1-m.alphaK) * vals['s2']/(1-vals['s2'])         ,
        0.0
        ])

        epsS_LR = supply_vec @ x0
        epsS_SR = m.delta * (1 - m.theta) * (supply_vec @ x_theta)
        # epsS_SR = m.delta * (supply_vec @ x_theta) # paperlike
        epsD    = - (-1/(1-m.alphaK) + demand_vec @ x0)

        print("\n" + "-"*44)
        print(" Elasticities ")
        print("-"*44)
        print(f"{'epsS_LR':<10} {epsS_LR:>8.2f}")
        print(f"{'epsS_SR':<10} {epsS_SR:>8.2f}")
        print(f"{'epsD':<10} {epsD:>8.2f}")

        # ========== ========== ========== ========== ========== 
        # tax elasticities
        price_elas  =  1 / (1-m.alphaK) * 1 / (epsS_LR + epsD)
        quant_elas  =  1 / (1-m.alphaK) * epsS_LR / (epsS_LR + epsD)
        wealth_elas = price_elas + quant_elas

        print("\n" + "-"*44)
        print(" Tax elasticities (LR GE) ")
        print("-"*44)
        print(f"{'price_elas':<10} {price_elas:>8.2f}")
        print(f"{'quant_elas':<10} {quant_elas:>8.2f}")
        print(f"{'wealth_ela':<10} {wealth_elas:>8.2f}")
        print("="*44 + "\n")

        # return central objects
        return {
            # incidence shares
            "consump_share": consump_w,
            "invest_share":  investm_w,
            "capital_share": capital_o,

            # key elasticities
            "epsS_LR": epsS_LR,
            "epsS_SR": epsS_SR,
            "epsD":    epsD,

            # GE tax elasticities
            "price_elas":  price_elas,
            "quant_elas":  quant_elas,
            "wealth_elas": wealth_elas,

            # optional: useful internals for debugging/reuse
            "ss": ss,
            "vals": vals,
            "x0": x0,
            "x_theta": x_theta,
        }




