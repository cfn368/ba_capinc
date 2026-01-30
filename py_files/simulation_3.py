import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests
import pandas as pd



def plot_figure5_emp(m, T=25, tau_ss=0.32, tauT=0.22, rho=0.7, ref=False, tail=50):
    # 1) steady state at initial tax (baseline you deviate from)
    ss = m.solve_steady_state(tau=tau_ss)

    # 2) build longer path, solve on longer horizon
    T_solve = int(T + tail)
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT,1.0/"
        "DNK.A..ST..S13+S1311+S13M.."
        "?startPeriod=2000&endPeriod=2025"
        "&dimensionAtObservation=AllDimensions"
        "&format=csvfilewithlabels"
    )

    r = requests.get(url)

    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    # keep only the statutory CIT rate
    df_cit = (df.loc[df["MEASURE"] == "CIT", ["TIME_PERIOD", "OBS_VALUE"]]
                .rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "cit_rate"})
                .assign(year=lambda x: x["year"].astype(int))
                .sort_values("year")
                .reset_index(drop=True))

    cit_pct = df_cit["cit_rate"].to_numpy(dtype=float) 
    tau_raw = cit_pct / 100.0 
    tau_t = np.concatenate([tau_raw, np.full(tail, tau_raw[-1])])
    net_t = 1.0 - tau_t
    net0 = net_t[0]
    dlog = np.log(net_t / net0)

    # terminal closure at tauT
    sim_long = m.solve_transition(tau_path=tau_t, tau_terminal=tauT)

    # 3) truncate for plotting (and welfare calculations)
    sl = slice(0, T + 1)
    sim = {k: (np.asarray(v)[sl] if isinstance(v, (list, np.ndarray)) and len(v) == T_solve + 1 else v)
           for k, v in sim_long.items()}
    dlog_net = np.asarray(dlog)[sl]
    # net_t = np.asarray(net_long)[sl]
    tau_t = np.asarray(tau_t)[sl]

    h = np.arange(T + 1)

    # 4) percent deviations (panel a) vs initial SS
    pct = lambda x, xss: 100 * np.log(np.asarray(x) / float(xss))
    dq  = pct(sim["q"],  ss["q"])
    dpI = pct(sim["pI"], ss["pI"])
    dK  = pct(sim["K"],  ss["K"])

    # 5) welfare block (unchanged, but uses truncated sim + dlog_net)
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

    return fig, (ax1, ax2), ss, sim, {"tau_t": tau_t, "dlog_net": dlog_net}

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