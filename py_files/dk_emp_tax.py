import numpy as np
import matplotlib.pyplot as plt

def net_tax_path_power(T=25, tau0=0.32, tauT=0.22, kappa=0.4):
    """
    Smooth monotone path from tau0 to tauT over T horizons.
    Convex in 'progress' if kappa>1 (slow early, fast late).
    Interpolation is done in log(1-tau) space.
    """
    t = np.arange(T + 1)
    s = (t / T) ** kappa  # progress in [0,1]

    net0 = 1.0 - tau0
    netT = 1.0 - tauT

    log_net = (1.0 - s) * np.log(net0) + s * np.log(netT)
    net_t = np.exp(log_net)
    tau_t = 1.0 - net_t

    # define "steady state" as the endpoint (like a transition to a new SS)
    dlog = np.log(net_t) - np.log(netT)
    return net_t, tau_t, dlog