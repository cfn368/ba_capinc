from scipy.optimize import root
from scipy.optimize import least_squares
import numpy as np


class CapIncModel_single:

    # 1. initiate
    def __init__(self):

        self.r          = 0.07  # world interest rate
        self.delta      = 0.15  # depreciation rate
        self.theta      = 0.25  # capital share in capital production
        
        # self.alphaK     = 0.36  
        # self.alpha1L    = 0.39  
        # self.alpha2L    = 0.25  
        # self.betaK      = 0.26  
        # self.beta1L     = 0.46 
        # self.beta2L     = 0.28  
        
        # DK calib (WIP)
        # main: 0.609, 0.647, 0.65
        # 1970: 0.542, 0.645, 0.8
        # 2020: 0.595, 0.641, 0.6
        
        self.alphaL     = 0.609    # 2022
        self.alphaK     = 1- self.alphaL
        
        self.betaL      = 0.647    # 2022
        self.betaK      = 1 - self.betaL
        
        self.mu         = 0.26  # labour adjustment cost param
        self.phi        = 0.65  # specialised labour supply: 0.8, 0.6 main: 0.7
        self.L          = 1.0   
        self.z_last     = np.array([0.0, 0.0, 0.0])
        self._ss        = None

    # ========== ========== ========== ========== ==========
    # 2. capital accumulation
    def next_K(self, K, I):
        # 3.2 capital accumulation: K'=(1-δ)K + K^θ I^(1-θ)
        return (1 - self.delta) * K + (K ** self.theta) * (I ** (1 - self.theta))

    # ========== ========== ========== ========== ==========
    # 3. static block: solve (sK,sL,pI) given (K,q,τ)
    def static_block_sigmoid(self, K, q, tau, z0=(0.0, 0.0, 0.0)):

        # 3.1 map unconstrained z to shares in (0,1)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # 3.2 parameters (local names = shorter formulas)
        L = self.L
        aK, aL = self.alphaK, self.alphaL
        bK, bL = self.betaK, self.betaL
        mu, phi = self.mu, self.phi
        theta = self.theta

        def F(z):
            # 3.3 unknowns: shares + log price (pI>0)
            zK, zL, u = z
            sK = sigmoid(zK); sL = sigmoid(zL) 
            pI = np.exp(u)

            # 3.4 sectoral allocations implied by shares
            KC, KI = (1 - sK) * K,  sK * K
            LC, LI = (1 - sL) * L, sL * L

            # 3.5 outputs (C-sector numeraire; I-sector value uses pI)
            C = (KC**aK) * (LC**aL) 
            I = (KI**bK) * (LI**bL) 

            # 3.6 marginal products / factor prices by Cobb–Douglas
            wC = aL * C / LC
            wI = bL * pI * I / LI

            rC = aK * C / KC        
            rI = bK * pI * I / KI   

            # 3.7 equilibrium conditions
            eq1 = rC - rI
            eq2 = (LC / LI) - ((1 - mu) / mu) * (wC / wI) ** phi

            # 3.8 pin down pI via the I/K target implied by (q,pI,θ)
            target_I_over_K = (1 - theta) ** (1 / theta) * (q / pI) ** (1 / theta)
            eq3 = (I / K) - target_I_over_K

            return np.array([eq1, eq2, eq3], float)

        # 3.9 solve static system (warm-start with z0)
        sol = root(F, np.array(z0, float), method="hybr")
        res = F(sol.x)

        # 3.10 decode solution and recompute objects cleanly
        zK, zL, u = sol.x
        sK, sL = sigmoid(zK), sigmoid(zL)
        pI = float(np.exp(u))

        KC, KI   = (1 - sK) * K,  sK * K
        LC, LI = (1 - sL) * L, sL * L
        C = (KC**aK) * (LC**aL) 
        I = (KI**bK) * (LI**bL) 
        wC = aL * C / LC
        wI = bL * pI * I / LI

        # 3.11 gross MPK in C-sector (before tax); tax enters only in Euler later
        rC_gross = aK * C / KC

        return {
            "sK": sK, "sL": sL, "pI": pI,
            "C": C, "I": I, "rC_gross": rC_gross,
            "success": sol.success, "message": sol.message,
            "z_sol": sol.x,
            "residuals": res,
            "max_abs_resid": float(np.max(np.abs(res))),
            "mins": {"KC": KC, "KI": KI, "LC": LC, "LI": LI},
            "wC": wC, "wI": wI,
            "LC": LC, "LI": LI
        } 

    # 4.12 thin wrapper: (i) warm-start with last z, (ii) fail fast if solver fails
    def _static(self, K, q, tau):
        
        out = self.static_block_sigmoid(K, q, tau, z0=self.z_last)

        if not out["success"]:
            raise RuntimeError(f"Static block failed: {out['message']}")
        self.z_last = out["z_sol"].copy()
        return out


    # ========== ========== ========== ========== ==========
    # 5. steady state: solve for (K,q) such that (i) K'=K and (ii) SS Euler holds
    def solve_steady_state(self, K_guess=1.0, q_guess=1.0, tau=0.0):
        r, delta, theta = self.r, self.delta, self.theta
        tau = float(tau)

        def G(x):
            logK, logq = x
            K, q = np.exp(logK), np.exp(logq)

            # reset warm-start so SS solve does not depend on previous path calls
            z0_old = self.z_last.copy()
            self.z_last[:] = 0.0
            st = self._static(K, q, tau=tau) 
            self.z_last = z0_old

            I = st["I"]
            K_next = self.next_K(K, I)

            mpk_term = 1 - delta + theta * (I / K) ** (1 - theta)

            # SS Euler / asset pricing for installed capital
            euler = (1 + r) * q - ((1 - tau) * st["rC_gross"] + q * mpk_term)
            return np.array([K_next - K, euler], float)

        sol = root(G, np.array([np.log(K_guess), np.log(q_guess)], float), method="hybr")
        if not sol.success:
            raise RuntimeError(f"SS solve failed: {sol.message}")

        K_ss, q_ss = np.exp(sol.x[0]), np.exp(sol.x[1])

        # store SS objects (also resets warm-start for a clean evaluation)
        self.z_last[:] = 0.0
        st = self._static(K_ss, q_ss, tau=tau)

        ss = {
            "K": K_ss, "q": q_ss, "pI": st["pI"], "I": st["I"], "C": st["C"],
            "sK": st["sK"], "sL": st["sL"], "rC_gross": st["rC_gross"],
            "tau": tau,                     
        }
        self._ss = ss
        return ss


    # ========== ========== ========== ========== ========== 
    # 6. given a candidate q-path, simulate (K,I,C,pI,rC,shares) forward
    def _path_given_q(self, q_path, K0, tau_path):
        # 6.1 allocate arrays for the implied path
        T  = len(q_path) - 1
        K  = np.empty(T + 1)
        pI = np.empty(T + 1)
        I  = np.empty(T + 1)
        C  = np.empty(T + 1)
        rC = np.empty(T + 1)
        sK = np.empty(T + 1)
        sL = np.empty(T + 1)
        wC = np.empty(T + 1) 
        wI = np.empty(T + 1) 
        LC = np.empty(T + 1)
        LI = np.empty(T + 1)

        # 6.2 initial condition
        K[0] = K0

        # 6.3 restart warm-start at t=0, then carry it forward across t (stability + speed)
        self.z_last[:] = 0.0

        for t in range(T + 1):
            # 6.4 solve static allocations at (K_t,q_t,τ_t)
            st = self._static(K[t], float(q_path[t]), float(tau_path[t]))
            pI[t] = st["pI"]
            I[t]  = st["I"]
            C[t]  = st["C"]
            rC[t] = st["rC_gross"]
            sK[t] = st["sK"]
            sL[t] = st["sL"] 
            
            wC[t] = st["wC"]
            wI[t] = st["wI"]
            LC[t] = st["LC"]
            LI[t] = st["LI"]

            # 6.5 update capital using the law of motion
            if t < T:
                K[t + 1] = self.next_K(K[t], I[t])

        return {"K": K, "q": q_path, "pI": pI, "I": I, "C": C, "rC_gross": rC,
                "sK": sK, "sL": sL,
                "wC": wC, "wI": wI,
                "LC": LC, "LI": LI}


    # ========== ========== ========== ========== ==========
    # 7. perfect foresight: choose q_path so Euler holds each t plus terminal condition
    def solve_transition(self, tau_path, K0=None, q_guess_path=None, method="krylov", tau_terminal=None):
        # ensure SS
        if self._ss is None:
            self.solve_steady_state()
        ss = self._ss

        T = len(tau_path) - 1
        if K0 is None:
            K0 = ss["K"]
        if q_guess_path is None:
            q_guess_path = ss["q"] * np.ones(T + 1)

        # terminal policy (if not provided, just hold last tau fixed)
        if tau_terminal is None:
            tauTplus = float(tau_path[-1])
        else:
            tauTplus = float(tau_terminal)

        # terminal continuation value q_{T+1}
        ssT = self.solve_steady_state(tau=tauTplus)  # uses your existing SS solver
        q_terminal = ssT["q"]

        def R(q_vec):
            q_vec = np.asarray(q_vec, float)          # unknowns: q_0,...,q_T  (length T+1)

            # extend paths by one extra terminal period
            q_ext   = np.concatenate([q_vec,  [q_terminal]])         # length T+2
            tau_ext = np.concatenate([tau_path, [tauTplus]])         # length T+2

            sim = self._path_given_q(q_ext, K0, tau_ext)
            K, I, rC = sim["K"], sim["I"], sim["rC_gross"]

            res = np.empty(T + 1)

            # impose Euler for t=0,...,T  (note the last one uses q_{T+1}=q_terminal)
            for t in range(T + 1):
                mpk_term_next = 1 - self.delta + self.theta * (I[t + 1] / K[t + 1]) ** (1 - self.theta)
                res[t] = (1 + self.r) * q_ext[t] - (
                    (1 - tau_ext[t + 1]) * rC[t + 1] + q_ext[t + 1] * mpk_term_next
                )

            return res

        sol = root(R, np.asarray(q_guess_path, float), method=method)
        if not sol.success:
            raise RuntimeError(f"Transition solve failed: {sol.message}")

        q_path = np.asarray(sol.x if hasattr(sol, "x") else q_guess_path, float)
        sim = self._path_given_q(q_path, K0, tau_path)

        sim["tau"] = np.asarray(tau_path, float)
        sim["success"] = bool(getattr(sol, "success", False))
        sim["message"] = str(getattr(sol, "message", "no message"))
        return sim
    


    # ========== ========== ========== ========== ========== 
    # Calibrate (mu, phi)
    #  - enforce zero SS wage premia
    #  - match investment-sector labor supply elasticities:
    #       (1 - sL)*phi = target_elas
    def calibrate(
        self,
        *,
        tau=0.0,
        K_guess=1.0,
        q_guess=1.0,
        target_elas=1.0,
        clip=1e-10,
        verbose=True,
    ):
        tau = float(tau)

        # keep old values in case something fails
        mu_old = float(self.mu)
        phi_old = float(self.phi)

        def _sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        def _logit(p):
            p = np.clip(float(p), clip, 1.0 - clip)
            return np.log(p / (1.0 - p))

        def _premia_and_elas():
            # steady state + static eval at SS
            ss = self.solve_steady_state(K_guess=K_guess, q_guess=q_guess, tau=tau)
            st = self.static_block_sigmoid(
                K=ss["K"], q=ss["q"], tau=tau, z0=(0.0, 0.0, 0.0)
            )

            prem = float(np.log(st["wI"] / st["wC"]))

            # implied investment-sector labor supply elasticities
            sL = float(st["sL"]) 
            eps = (1.0 - sL) * float(self.phi)

            return ss, st, prem, eps

        try:
            # unknowns: z = [logit(mu), log(phi)]
            z0 = np.array(
                [
                    _logit(mu_old),
                    np.log(max(phi_old, 1e-12)),
                ],
                float,
            )

            def H(z):
                self.mu = _sigmoid(z[0])
                self.phi = float(np.exp(z[1]))

                _, _, prem, eps = _premia_and_elas()

                return np.array(
                    [
                        prem,                     # = 0
                        eps - float(target_elas), # = 0
                    ],
                    float,
                )

            sol = root(H, z0, method="hybr")
            if not sol.success:
                raise RuntimeError(sol.message)

            # decode solution cleanly
            self.mu = _sigmoid(sol.x[0])
            self.phi = float(np.exp(sol.x[1]))

            ss, st, prem, eps = _premia_and_elas()

            if verbose:
                # implied by old params (restore temporarily)
                self.mu, self.phi = mu_old, phi_old
                _, _, prem_old, eps_old = _premia_and_elas()

                # restore new
                self.mu = _sigmoid(sol.x[0])
                self.phi = float(np.exp(sol.x[1]))

                print("\n" + "=" * 60)
                print(" Calibrate household: zero wage premia + target eps_nI ")
                print("=" * 60)
                print(f"{'targets':<10} prem=0, prem=0, eps={target_elas:.1f}")
                print("-" * 60)
                print(f"{'old':<10} mu={mu_old:.2f}   =>")
                print(f"{'':<10} log(wI/wC)={prem_old:+.2e}\n")

                print(f"{'old':<10} phi={phi_old:.2f} =>")
                print(f"{'':<10} eps={eps_old:.3f}")
                print("-" * 60)
                print(f"{'new':<10} mu={float(self.mu):.4f}   =>")
                print(f"{'':<10} log(wI/wC)={prem:+.2e}\n")
                
                print(f"{'new':<10} phi={float(self.phi):.4f} =>")
                print(f"{'':<10} eps={eps:.3f}")

                print("=" * 60 + "\n")

            return {
                "mu": float(self.mu),
                "phi": float(self.phi),
                "prem_log_wI_wC": prem,
                "eps_I": float(eps),
                "ss": ss,
                "static_at_ss": st,
                "success": True,
            }

        except Exception as e:
            self.mu, self.phi = mu_old, phi_old
            raise RuntimeError(f"household calibration failed: {e}")

