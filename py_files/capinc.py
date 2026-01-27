from scipy.optimize import root
import numpy as np


class CapIncModel:

    # 1. initiate
    def __init__(self):

        self.r          = 0.07  # world interest rate
        self.delta      = 0.15  # depreciation rate
        self.theta      = 0.27  # capital share in capital production
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

    # ========== 
    # 2. shock path: shock is to log(1-τ_t) 
    def shock_dlog_net_tax(self, T=25, size=0.01, decay=0.10):
        # 2.1 dlog(1-τ_t) = size*(1-decay)^t
        t = np.arange(T + 1)
        return size * (1 - decay) ** t

    def net_tax_path(self, T=25, tau_ss=0.0, size=0.01, decay=0.10):
        # 2.2 build (1-τ_t) from steady state net-of-tax, then back out τ_t
        dlog = self.shock_dlog_net_tax(T=T, size=size, decay=decay)
        net_ss = 1.0 - tau_ss
        net_t  = net_ss * np.exp(dlog)
        tau_t  = 1.0 - net_t
        return net_t, tau_t, dlog

    # ========== 
    # 3. primitives used throughout
    def investment(self, K, q, pI):
        # 3.1 target I implied by the q vs pI condition: I/K = (1-θ)^(1/θ) (q/pI)^(1/θ)
        return K * (1 - self.theta) ** (1 / self.theta) * (q / pI) ** (1 / self.theta)

    def next_K(self, K, I):
        # 3.2 capital accumulation: K'=(1-δ)K + K^θ I^(1-θ)
        return (1 - self.delta) * K + (K ** self.theta) * (I ** (1 - self.theta))


    # ========== 
    # 4. static block: solve (sK,s1,s2,pI) given (K,q,τ)
    def static_block_sigmoid(self, K, q, tau, z0=(0.0, 0.0, 0.0, 0.0)):

        # 4.1 map unconstrained z to shares in (0,1)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # 4.2 parameters (local names = shorter formulas)
        L1, L2 = self.L1, self.L2
        aK, a1, a2 = self.alphaK, self.alpha1L, self.alpha2L
        bK, b1, b2 = self.betaK, self.beta1L, self.beta2L
        mu1, mu2, phi1, phi2 = self.mu1, self.mu2, self.phi1, self.phi2
        theta = self.theta

        def F(z):
            # 4.3 unknowns: shares + log price (pI>0)
            zK, z1, z2, u = z
            sK = sigmoid(zK); s1 = sigmoid(z1); s2 = sigmoid(z2)
            pI = np.exp(u)

            # 4.4 sectoral allocations implied by shares
            KC, KI   = (1 - sK) * K,  sK * K
            L1C, L1I = (1 - s1) * L1, s1 * L1
            L2C, L2I = (1 - s2) * L2, s2 * L2

            # 4.5 outputs (C-sector numeraire; I-sector value uses pI)
            C = (KC**aK) * (L1C**a1) * (L2C**a2)
            I = (KI**bK) * (L1I**b1) * (L2I**b2)

            # 4.6 marginal products / factor prices by Cobb–Douglas
            w1C = a1 * C / L1C
            w2C = a2 * C / L2C
            w1I = b1 * pI * I / L1I
            w2I = b2 * pI * I / L2I

            rC = aK * C / KC
            rI = bK * pI * I / KI

            # 4.7 equilibrium conditions
            eq1 = rC - rI
            eq2 = (L1C / L1I) - ((1 - mu1) / mu1) * (w1C / w1I) ** phi1
            eq3 = (L2C / L2I) - ((1 - mu2) / mu2) * (w2C / w2I) ** phi2

            # 4.8 pin down pI via the I/K target implied by (q,pI,θ)
            target_I_over_K = (1 - theta) ** (1 / theta) * (q / pI) ** (1 / theta)
            eq4 = (I / K) - target_I_over_K

            return np.array([eq1, eq2, eq3, eq4], float)

        # 4.9 solve static system (warm-start with z0)
        sol = root(F, np.array(z0, float), method="hybr")
        res = F(sol.x)

        # 4.10 decode solution and recompute objects cleanly
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

        # 4.11 gross MPK in C-sector (before tax); tax enters only in Euler later
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

    def _static(self, K, q, tau):
        # 4.12 thin wrapper: (i) warm-start with last z, (ii) fail fast if solver fails
        out = self.static_block_sigmoid(K, q, tau, z0=self.z_last)
        if not out["success"]:
            raise RuntimeError(f"Static block failed: {out['message']}")
        self.z_last = out["z_sol"].copy()
        return out


    # ========== 
    # 5. steady state: solve for (K,q) such that (i) K'=K and (ii) SS Euler holds
    def solve_steady_state(self, K_guess=1.0, q_guess=1.0):
        r, delta, theta = self.r, self.delta, self.theta

        def G(x):
            # 5.1 unknowns in logs => keeps K,q positive
            logK, logq = x
            K, q = np.exp(logK), np.exp(logq)

            # 5.2 reset warm-start so SS solve does not depend on previous path calls
            z0_old = self.z_last.copy()
            self.z_last[:] = 0.0
            st = self._static(K, q, tau=0.0)
            self.z_last = z0_old

            # 5.3 SS capital stationarity
            I = st["I"]
            K_next = self.next_K(K, I)

            # 5.4 ∂K'/∂K term from accumulation technology
            mpk_term = 1 - delta + theta * (I / K) ** (1 - theta)

            # 5.5 SS Euler / asset pricing for installed capital
            euler = (1 + r) * q - (st["rC_gross"] + q * mpk_term)

            return np.array([K_next - K, euler], float)

        # 5.6 solve the 2x2 SS system
        sol = root(G, np.array([np.log(K_guess), np.log(q_guess)], float), method="hybr")
        if not sol.success:
            raise RuntimeError(f"SS solve failed: {sol.message}")

        K_ss, q_ss = np.exp(sol.x[0]), np.exp(sol.x[1])

        # 5.7 store SS objects (also resets warm-start for a clean evaluation)
        self.z_last[:] = 0.0
        st = self._static(K_ss, q_ss, tau=0.0)

        ss = {
            "K": K_ss, "q": q_ss, "pI": st["pI"], "I": st["I"], "C": st["C"],
            "sK": st["sK"], "s1": st["s1"], "s2": st["s2"], "rC_gross": st["rC_gross"],
        }
        self._ss = ss
        return ss


    # ========== 
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

    # 7. perfect foresight: choose q_path so Euler holds each t plus terminal condition
    def solve_transition(self, tau_path, K0=None, q_guess_path=None, method="krylov"):
        # 7.1 ensure we have SS for terminal condition and default guesses
        if self._ss is None:
            self.solve_steady_state()

        ss = self._ss
        T = len(tau_path) - 1

        # 7.2 default initial capital = SS capital
        if K0 is None:
            K0 = ss["K"]

        # 7.3 default initial guess for q-path = flat at SS q
        if q_guess_path is None:
            q_guess_path = ss["q"] * np.ones(T + 1)

        def R(q_vec):
            # 7.4 for a candidate q-path, simulate quantities and build Euler residuals
            q_vec = np.asarray(q_vec, float)
            sim = self._path_given_q(q_vec, K0, tau_path)
            K, I, rC = sim["K"], sim["I"], sim["rC_gross"]

            res = np.empty(T + 1)

            # 7.5 Euler / arbitrage for t=0..T-1 (tax hits next period payoff)
            for t in range(T):
                mpk_term_next = 1 - self.delta + self.theta * (I[t + 1] / K[t + 1]) ** (1 - self.theta)
                res[t] = (1 + self.r) * q_vec[t] - (
                    (1 - tau_path[t + 1]) * rC[t + 1] + q_vec[t + 1] * mpk_term_next
                )

            # 7.6 terminal: force q_T back to SS q (finite-horizon closure)
            # 7.6 terminal: force q_T back to SS q (finite-horizon closure)
            if tau_terminal is None:
                qT_target = ss["q"]
            else:
                ssT = self.solve_steady_state(tau=float(tau_terminal))
                qT_target = ssT["q"]

            res[T] = q_vec[T] - qT_target
            return res

            # res[T] = q_vec[T] - ss["q"]
            return res

        # 7.7 solve for the q-path
        sol = root(R, np.asarray(q_guess_path, float), method=method)

        # 7.8 fallback if krylov fails
        if not sol.success:
            sol2 = root(R, np.asarray(q_guess_path, float), method="hybr")
            if not sol2.success:
                raise RuntimeError(f"Transition solve failed: {sol.message} | fallback: {sol2.message}")
            sol = sol2

        # 7.9 return the implied path at the solution q-path
        sim = self._path_given_q(sol.x, K0, tau_path)
        sim["tau"] = np.asarray(tau_path, float)
        sim["success"] = True
        sim["message"] = sol.message
        return sim