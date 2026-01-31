import numpy as np
from io import StringIO
import requests
import pandas as pd

###########################################################
# 1. temporary tax cut
###########################################################

def temp_tc(T=25, tau_ss=0.0, size=0.01, 
            decay=0.10, tail=50, tau_terminal=None
            ):

    # 1.1 shock path helpers
    def shock_dlog_net_tax(T=25, size=0.01, decay=0.10):
        t = np.arange(T + 1)
        return size * (1 - decay) ** t

    def net_tax_path(T=25, tau_ss=0.0, size=0.01, decay=0.10):

        dlog = shock_dlog_net_tax(T=T, size=size, decay=decay)
        net_ss = 1.0 - tau_ss
        net_t  = net_ss * np.exp(dlog)
        tau_t  = 1.0 - net_t
        return net_t, tau_t, dlog

    # 1.2 shock path
    T_solve = int(T + tail)
    net_long, tau_long, dlog_net_long = net_tax_path(
        T=T_solve, 
        tau_ss=tau_ss, 
        size=size, 
        decay=decay
    )

    # 1.3 terminalize
    tauT = tau_ss if tau_terminal is None else float(tau_terminal)

    return net_long, tau_long, dlog_net_long, tauT
    

###########################################################
# 2. permanent tax cut
###########################################################

def perm_tc(T=25, tau0=0.32, tauT=0.22, rho=0.7, tail=50):
    T_solve = int(T + tail)
    t = np.arange(T_solve + 1, dtype=float)

    net0 = 1.0 - tau0
    netT = 1.0 - tauT

    net_path = netT + (rho ** t) * (net0 - netT)

    net_t = net_path.copy()
    net_t[T + 1:] = net_path[T]  
    tau_t = 1.0 - net_t
    dlog  = np.log(net_t / net0)

    return net_t, tau_t, dlog, tauT 

###########################################################
# 3. permanent tax cut (empirical)
###########################################################


def perm_tc_emp(tail=50):

    # 1. 25 year of data available, fetch
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

    # 2. keep only the statutory CIT rate
    df_cit = (df.loc[df["MEASURE"] == "CIT", ["TIME_PERIOD", "OBS_VALUE"]]
                .rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "cit_rate"})
                .assign(year=lambda x: x["year"].astype(int))
                .sort_values("year")
                .reset_index(drop=True))
    
    # 3. make output consistent
    cit_pct = df_cit["cit_rate"].to_numpy(dtype=float) 
    tau_raw = cit_pct / 100.0 
    tau_t = np.concatenate([tau_raw, np.full(tail, tau_raw[-1])])
    net_t = 1.0 - tau_t
    net0 = net_t[0]
    dlog = np.log(net_t / net0)
    tauT = tau_t[-1]

    return net_t, tau_t, dlog, tauT