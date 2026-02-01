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

        # total welfare-flow path
        WG_total_path = D_t * np.asarray(dlog_net, float)

        # residual claimant
        wg_K = WG_total_path - wg_L_path    

        # 4.5 ... in ss consumption units
        wC_pct = 100 * wg_C / ss["C"]
        wI_pct = 100 * wg_I / ss["C"]
        wK_pct = 100 * wg_K / ss["C"]
        WG_pct = 100 * WG_total_path / ss['C']

        # 4.6 append solution dict
        sim["dq_pct"]  = np.asarray(dq, float)
        sim["dpI_pct"] = np.asarray(dpI, float)
        sim["dK_pct"]  = np.asarray(dK, float)

        sim["wg_C"]  = np.asarray(wg_C, float)
        sim["wg_I"]  = np.asarray(wg_I, float)
        sim["wg_K"]  = np.asarray(wg_K, float)
        sim["WG_total_path"]  = np.asarray(WG_total_path, float)

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
        ax2.plot(h, WG_pct, lw=2, label="Total wealth gain", color='gray', ls=':')
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
# 2. incidence and elasticities
###########################################################

def inc_elas(m, sim, tau):

        # ========== ========== ========== ========== ========== 
        # 1. tax incidence
        consump_w = sim['wg_C'].sum() / sim['WG_total_path'].sum()
        investm_w = sim['wg_I'].sum() / sim['WG_total_path'].sum()
        capital_o = sim['wg_K'].sum() / sim['WG_total_path'].sum()

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
                vals['s1']/(1-vals['s1']) * (m.alpha1L-(1-m.phi1)/m.phi1)
                + m.beta1L - (1-m.phi1)/m.phi1                          ,
                vals['s2']/(1-vals['s2']) * m.alpha2L + m.beta2L        ,
                1.0
        ]   ,
        [
                vals['sK']/(1-vals['sK']) * m.alphaK + m.betaK          ,
                vals['s1']/(1-vals['s1']) * m.alpha1L + m.beta1L        ,
                vals['s2']/(1-vals['s2']) * (m.alpha2L-(1-m.phi2)/m.phi2)
                + m.beta2L - (1-m.phi2)/m.phi2                          ,
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




