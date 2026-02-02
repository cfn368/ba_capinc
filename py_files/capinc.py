from scipy.optimize import root
from scipy.optimize import least_squares
import numpy as np


class CapIncModel:

    # 1. initiate
    def __init__(self):

        self.r          = 0.07  # world interest rate
        self.delta      = 0.15  # depreciation rate
        self.theta      = 0.25  # capital share in capital production
        self.alphaK     = 0.36  # capital share in consumption production
        self.alpha1L    = 0.39  # labour 1 share in consumption production
        self.alpha2L    = 0.25  # labour 2 share in consumption production
        self.betaK      = 0.26  # capital share in investment production
        self.beta1L     = 0.46  # labour 1 share in investment production
        self.beta2L     = 0.28  # labour 2 share in investment production
        self.mu1        = 0.26  # labour 1 adjustment cost param
        self.mu2        = 0.25  # labour 2 adjustment cost param
        self.phi1       = 1.35  # labour 1 adjustment cost curvature
        self.phi2       = 0.40  # labour 2 adjustment cost curvature
        self.L1         = 1.0   # total labour 1 supply
        self.L2         = 1.0   # total labour 2 supply
        self.z_last     = np.array([0.0, 0.0, 0.0, 0.0])
        self._ss        = None

    # ========== ========== ========== ========== ==========
    # 2. capital accumulation
    def next_K(self, K, I):
        # 3.2 capital accumulation: K'=(1-δ)K + K^θ I^(1-θ)
        return (1 - self.delta) * K + (K ** self.theta) * (I ** (1 - self.theta))

    # ========== ========== ========== ========== ==========
    # 3. static block: solve (sK,s1,s2,pI) given (K,q,τ)
    def static_block_sigmoid(self, K, q, tau, z0=(0.0, 0.0, 0.0, 0.0)):

        # 3.1 map unconstrained z to shares in (0,1)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # 3.2 parameters (local names = shorter formulas)
        L1, L2 = self.L1, self.L2
        aK, a1, a2 = self.alphaK, self.alpha1L, self.alpha2L
        bK, b1, b2 = self.betaK, self.beta1L, self.beta2L
        mu1, mu2, phi1, phi2 = self.mu1, self.mu2, self.phi1, self.phi2
        theta = self.theta

        def F(z):
            # 3.3 unknowns: shares + log price (pI>0)
            zK, z1, z2, u = z
            sK = sigmoid(zK); s1 = sigmoid(z1); s2 = sigmoid(z2)
            pI = np.exp(u)

            # 3.4 sectoral allocations implied by shares
            KC, KI   = (1 - sK) * K,  sK * K
            L1C, L1I = (1 - s1) * L1, s1 * L1
            L2C, L2I = (1 - s2) * L2, s2 * L2

            # 3.5 outputs (C-sector numeraire; I-sector value uses pI)
            C = (KC**aK) * (L1C**a1) * (L2C**a2)
            I = (KI**bK) * (L1I**b1) * (L2I**b2)

            # 3.6 marginal products / factor prices by Cobb–Douglas
            w1C = a1 * C / L1C
            w2C = a2 * C / L2C
            w1I = b1 * pI * I / L1I
            w2I = b2 * pI * I / L2I

            rC = aK * C / KC        
            rI = bK * pI * I / KI   

            # 3.7 equilibrium conditions
            eq1 = rC - rI
            eq2 = (L1C / L1I) - ((1 - mu1) / mu1) * (w1C / w1I) ** phi1
            eq3 = (L2C / L2I) - ((1 - mu2) / mu2) * (w2C / w2I) ** phi2

            # 3.8 pin down pI via the I/K target implied by (q,pI,θ)
            target_I_over_K = (1 - theta) ** (1 / theta) * (q / pI) ** (1 / theta)
            eq4 = (I / K) - target_I_over_K

            return np.array([eq1, eq2, eq3, eq4], float)

        # 3.9 solve static system (warm-start with z0)
        sol = root(F, np.array(z0, float), method="hybr")
        res = F(sol.x)

        # 3.10 decode solution and recompute objects cleanly
        zK, z1, z2, u = sol.x
        sK, s1, s2 = sigmoid(zK), sigmoid(z1), sigmoid(z2)
        pI = float(np.exp(u))

        KC, KI   = (1 - sK) * K,  sK * K
        L1C, L1I = (1 - s1) * L1, s1 * L1
        L2C, L2I = (1 - s2) * L2, s2 * L2
        C = (KC**aK) * (L1C**a1) * (L2C**a2)
        I = (KI**bK) * (L1I**b1) * (L2I**b2)
        w1C = a1 * C / L1C
        w2C = a2 * C / L2C
        w1I = b1 * pI * I / L1I
        w2I = b2 * pI * I / L2I

        # 3.11 gross MPK in C-sector (before tax); tax enters only in Euler later
        rC_gross = aK * C / KC

        return {
            "sK": sK, "s1": s1, "s2": s2, "pI": pI,
            "C": C, "I": I, "rC_gross": rC_gross,
            "success": sol.success, "message": sol.message,
            "z_sol": sol.x,
            "residuals": res,
            "max_abs_resid": float(np.max(np.abs(res))),
            "mins": {"KC": KC, "KI": KI, "L1C": L1C, "L1I": L1I, "L2C": L2C, "L2I": L2I},
            "w1C": w1C, "w2C": w2C, "w1I": w1I, "w2I": w2I,
            "L1C": L1C, "L1I": L1I, "L2C": L2C, "L2I": L2I,
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
            st = self._static(K, q, tau=tau)   # <-- changed
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
        st = self._static(K_ss, q_ss, tau=tau)  # <-- changed

        ss = {
            "K": K_ss, "q": q_ss, "pI": st["pI"], "I": st["I"], "C": st["C"],
            "sK": st["sK"], "s1": st["s1"], "s2": st["s2"], "rC_gross": st["rC_gross"],
            "tau": tau,                          # <-- optional but helpful
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
        s1 = np.empty(T + 1)
        s2 = np.empty(T + 1)
        w1C = np.empty(T + 1); w2C = np.empty(T + 1)
        w1I = np.empty(T + 1); w2I = np.empty(T + 1)
        L1C = np.empty(T + 1); L1I = np.empty(T + 1)
        L2C = np.empty(T + 1); L2I = np.empty(T + 1)

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
            sK[t] = st["sK"]; s1[t] = st["s1"]; s2[t] = st["s2"]
            
            w1C[t] = st["w1C"]; w2C[t] = st["w2C"]
            w1I[t] = st["w1I"]; w2I[t] = st["w2I"]
            L1C[t] = st["L1C"]; L1I[t] = st["L1I"]
            L2C[t] = st["L2C"]; L2I[t] = st["L2I"]

            # 6.5 update capital using the law of motion
            if t < T:
                K[t + 1] = self.next_K(K[t], I[t])

        return {"K": K, "q": q_path, "pI": pI, "I": I, "C": C, "rC_gross": rC,
                "sK": sK, "s1": s1, "s2": s2,
                "w1C": w1C, "w2C": w2C, "w1I": w1I, "w2I": w2I,
                "L1C": L1C, "L1I": L1I, "L2C": L2C, "L2I": L2I}


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
    # Calibrate (mu1, mu2, phi1, phi2)
    #  - enforce zero SS wage premia
    #  - match investment-sector labor supply elasticities:
    #       (1 - s1)*phi1 = target_elas1
    #       (1 - s2)*phi2 = target_elas2
    def calibrate(
        self,
        *,
        tau=0.0,
        K_guess=1.0,
        q_guess=1.0,
        target_elas1=1.0,
        target_elas2=0.3,
        clip=1e-10,
        verbose=True,
    ):
        tau = float(tau)

        # keep old values in case something fails
        mu1_old, mu2_old = float(self.mu1), float(self.mu2)
        phi1_old, phi2_old = float(self.phi1), float(self.phi2)

        def _sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        def _logit(p):
            p = np.clip(float(p), clip, 1.0 - clip)
            return np.log(p / (1.0 - p))

        def _premia_and_elas():
            # steady state + static eval at SS
            ss = self.solve_steady_state(K_guess=K_guess, q_guess=q_guess, tau=tau)
            st = self.static_block_sigmoid(
                K=ss["K"], q=ss["q"], tau=tau, z0=(0.0, 0.0, 0.0, 0.0)
            )

            prem1 = float(np.log(st["w1I"] / st["w1C"]))
            prem2 = float(np.log(st["w2I"] / st["w2C"]))

            # implied investment-sector labor supply elasticities
            # eps_nI = (1 - s_n) * phi_n
            s1, s2 = float(st["s1"]), float(st["s2"])
            eps1 = (1.0 - s1) * float(self.phi1)
            eps2 = (1.0 - s2) * float(self.phi2)

            return ss, st, prem1, prem2, eps1, eps2

        try:
            # unknowns: z = [logit(mu1), logit(mu2), log(phi1), log(phi2)]
            z0 = np.array(
                [
                    _logit(mu1_old),
                    _logit(mu2_old),
                    np.log(max(phi1_old, 1e-12)),
                    np.log(max(phi2_old, 1e-12)),
                ],
                float,
            )

            def H(z):
                self.mu1 = _sigmoid(z[0])
                self.mu2 = _sigmoid(z[1])
                self.phi1 = float(np.exp(z[2]))
                self.phi2 = float(np.exp(z[3]))

                _, _, prem1, prem2, eps1, eps2 = _premia_and_elas()

                return np.array(
                    [
                        prem1,                      # = 0
                        prem2,                      # = 0
                        eps1 - float(target_elas1), # = 0
                        eps2 - float(target_elas2), # = 0
                    ],
                    float,
                )

            sol = root(H, z0, method="hybr")
            if not sol.success:
                raise RuntimeError(sol.message)

            # decode solution cleanly
            self.mu1 = _sigmoid(sol.x[0])
            self.mu2 = _sigmoid(sol.x[1])
            self.phi1 = float(np.exp(sol.x[2]))
            self.phi2 = float(np.exp(sol.x[3]))

            ss, st, prem1, prem2, eps1, eps2 = _premia_and_elas()

            if verbose:
                # implied by old params (restore temporarily)
                self.mu1, self.mu2, self.phi1, self.phi2 = mu1_old, mu2_old, phi1_old, phi2_old
                _, _, prem1_old, prem2_old, eps1_old, eps2_old = _premia_and_elas()

                # restore new
                self.mu1 = _sigmoid(sol.x[0])
                self.mu2 = _sigmoid(sol.x[1])
                self.phi1 = float(np.exp(sol.x[2]))
                self.phi2 = float(np.exp(sol.x[3]))

                print("\n" + "=" * 60)
                print(" Calibrate household: zero wage premia + target eps_nI ")
                print("=" * 60)
                print(f"{'targets':<10} prem1=0, prem2=0, eps1={target_elas1:.3f}, eps2={target_elas2:.3f}")
                print("-" * 60)
                print(f"{'old':<10} mu1={mu1_old:.4f}, mu2={mu2_old:.4f}, phi1={phi1_old:.4f}, phi2={phi2_old:.4f}")
                print(f"{'':<10} log(w1I/w1C)={prem1_old:+.2e}, log(w2I/w2C)={prem2_old:+.2e}, eps1={eps1_old:.3f}, eps2={eps2_old:.3f}")
                print("-" * 60)
                print(f"{'new':<10} mu1={float(self.mu1):.4f}, mu2={float(self.mu2):.4f}, phi1={float(self.phi1):.4f}, phi2={float(self.phi2):.4f}")
                print(f"{'':<10} log(w1I/w1C)={prem1:+.2e}, log(w2I/w2C)={prem2:+.2e}, eps1={eps1:.3f}, eps2={eps2:.3f}")
                print("=" * 60 + "\n")

            return {
                "mu1": float(self.mu1),
                "mu2": float(self.mu2),
                "phi1": float(self.phi1),
                "phi2": float(self.phi2),
                "prem1_log_wI_wC": prem1,
                "prem2_log_wI_wC": prem2,
                "eps1_I": float(eps1),
                "eps2_I": float(eps2),
                "ss": ss,
                "static_at_ss": st,
                "success": True,
            }

        except Exception as e:
            self.mu1, self.mu2, self.phi1, self.phi2 = mu1_old, mu2_old, phi1_old, phi2_old
            raise RuntimeError(f"household calibration failed: {e}")

