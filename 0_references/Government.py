from Laborer import labourer_class
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
from scipy import optimize


class GovernmentClass(labourer_class):

    # 1. initiate
    def __init__(self,par=None):
        
        # 1.1 load
        self.setup_worker()
        self.setup_government()

        # 1.2 update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # 1.3 random number generator
        self.rng = np.random.default_rng(2002)
        self.p_vec = self.draw_productivities()
        self.par.p_vec = self.p_vec
        
        
    # 2. set up government
    def setup_government(self):

        # 2.1 define parameters
        par = self.par
        par.N = 100  
        par.sigma_p = 0.3  
        par.chi = 1/2 * par.N
        par.eta = 0.1 
        par.kappa   = 9
        par.omega   = 0.2
        
        
    # 3. draw productivities
    def draw_productivities(self):

        par = self.par
        p = np.exp(self.rng.normal(
                loc=-0.5*par.sigma_p**2, 
                scale=par.sigma_p, 
                size=par.N)) 

        return p
        
        
    # 4. Total tax revenue
    def tot_tax_rev(self, return_micro=False, tau=0.5, zeta=0.1):
        
        par = self.par
        sol = self.sol
        par.tau = tau
        par.zeta = zeta
        
        # 4.1 initiate vectors
        ell_vec = np.empty_like(par.p_vec)
        
        # 4.2 solve workers prob
        for i, p_val in enumerate(par.p_vec):
            par.p = p_val
            self.root_finder(verbose=False)
            ell_vec[i] = self.sol.ell_star_root
            
        # 4.3 compute total tax rev
        T = par.N*zeta + sum(tau*par.w*par.p_vec*ell_vec)
        sol.ell_vec = ell_vec
        
        if return_micro:
            return T, ell_vec
        else:
            return T
        
        
    # 5. SWF
    def SWF(self, tau=0.5, zeta=0.1):
        
        par = self.par
        sol = self.sol
        par.tau = tau
        par.zeta = zeta
        
        T, ell_vec = self.tot_tax_rev(return_micro=True, tau=tau, zeta=zeta)
        U_vec = np.empty_like(ell_vec, dtype=float)
        
        # 5.1 compute worker utility
        for i, (p_val, ell_val) in enumerate(zip(par.p_vec, ell_vec)):
            par.p = p_val                
            U_vec[i] = self.utility(ell=ell_val)
        
        # 5.2 compute SWF-value
        SWF_val = par.chi * T**par.eta + np.sum(U_vec)
        
        return SWF_val
    
    
    # 6. maximize SWF
    def max_SWF(self, zeta=1, tau=0.5, verbose=True, x0=[0.1,0.1]):
        
        # 6.1 get params
        par = self.par
        sol = self.sol
        par.tau = tau
        par.zeta = zeta
        
        tau_min, tau_max = 0, 0.9999
        p_min = par.p_vec.min()
        
        # 6.2 define objective
        def obj(x):
            tau, zeta = x 
            return -self.SWF(tau=tau, zeta=zeta)
        
        # 6.3 build bounds (LL: ineq meakes sure eq is below upper bound)
        cons = [
            {
                "type": "ineq",
                "fun": lambda x: (1 - x[0]) * par.w * p_min * par.ell_max - x[1],
            },
            {
                "type": "ineq",
                "fun": lambda x: self.tot_tax_rev(tau=x[0], zeta=x[1]),
            },
        ]
        bounds = [(tau_min, tau_max), (None, None)]
        
        # 6.4 set up solver
        res = optimize.minimize(
            obj,
            x0=x0,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
        )
        
        # 6.5 get solution
        sol.tau_star, sol.zeta_star = res.x
        sol.SWF_max = -res.fun
        
        # 6.6 verify bounds for zetz
        zeta_max = (1-sol.tau_star) * par.w*p_min*par.ell_max
        
        if verbose:
            print("\n=== Numerical solution for optimal tax parameters ===\n")
            print(f"Optimal tax rate (tau*)    : {sol.tau_star:.4f}")
            print(f"Optimal Transfer (zeta*)   : {sol.zeta_star:.4f}")
            print("-----------------------------------------------")
            print(f"Maximum SWF                : {sol.SWF_max:.6f}")
            print(f"Converged?                 : {res.success}")
            print(f"Zeta upper bound           : {zeta_max:.4f} \n")
     
        
    # With TOP-TAX #####################################################################
        
        
    # 7. ell type (out of the three defined)
    def ell_type(self):
        
        par = self.par
        sol = self.sol
        
        # 7.1 use labour supply under TOP TAX
        ell_vec = sol.ell_vec_top_tax   # <-- changed line
        p_vec   = par.p_vec
        
        # 7.2 make type and threshold
        type_vec = np.empty_like(ell_vec, dtype='<U1')
        threshold_vec = par.kappa / (par.w * p_vec)
        
        # 7.3 fill type
        type_vec[ell_vec >  threshold_vec]            = 'a'
        type_vec[np.isclose(ell_vec, threshold_vec)]  = 'k'
        type_vec[ell_vec <  threshold_vec]            = 'b'

        return type_vec, threshold_vec
    
    
    # 8. redefine tot_tax_rev
    def tot_tax_rev_top_tax(self, return_micro=True, tau=0.5, zeta=0.1):
        
        par = self.par
        sol = self.sol
        par.tau = tau
        par.zeta = zeta

        ell_vec = np.empty_like(par.p_vec)
        
        # 8.1 solve workers prob
        for i, p_val in enumerate(par.p_vec):
            par.p = p_val
            self.solve_top_tax_root(verbose=False)
            ell_vec[i] = sol.ell_star_top
            
        # 8.2 revenue per worker and total tax
        y_gross = par.w * par.p_vec * ell_vec
        
        top_part = par.omega * np.maximum(y_gross - par.kappa, 0.0)
        T = par.N * zeta + np.sum(par.tau * y_gross + top_part)
        
        sol.ell_vec_top_tax = ell_vec   
         
        if return_micro:
            return T, ell_vec
        else:
            return T
    
    
    # 9. SWF_top
    def SWF_top(self, tau=0.5, zeta=0.1):
        
        par = self.par
        sol = self.sol
        par.tau = tau
        par.zeta = zeta
        
        # 9.1 get ell_vec
        T, ell_vec = self.tot_tax_rev_top_tax(return_micro=True, tau=tau, zeta=zeta)
        U_vec = np.empty_like(ell_vec, dtype=float)
        
        # 9.2 compute worker utility
        for i, (p_val, ell_val) in enumerate(zip(par.p_vec, ell_vec)):
            par.p = p_val                
            U_vec[i] = self.utility_top(ell=ell_val)
        
        # 9.3 compute SWF-value
        SWF_val = par.chi * T**par.eta + np.sum(U_vec)
        
        return SWF_val
    
    
    # 10. joint objective: maximize SWF with top-tax over (tau, zeta, omega, kappa)
    def obj_policies(self, x):
        
        par = self.par
        tau, zeta, omega, kappa = x

        tau_min, tau_max = 0.0, 0.9999
        p_min = par.p_vec.min()

        # 10.1 simple bound penalties
        if (tau < tau_min) or (tau >= tau_max):
            return 1e6

        # 10.2 top tax cannot exceed (1 - tau)
        if (omega < 0.0) or (omega >= 1.0 - tau):
            return 1e6

        # 10.3
        if kappa <= 0.0:
            return 1e6

        # 10.4 ensure zeta does not violate c(ell_max,p_min) ≥ 0
        zeta_max = (1.0 - tau) * par.w * p_min * par.ell_max
        if zeta > zeta_max:
            return 1e6

        # 10.2 set policy parameters
        par.tau   = tau
        par.zeta  = zeta
        par.omega = omega
        par.kappa = kappa

        # 10.3 ensure total tax revenue with top tax is non-negative
        T, _ = self.tot_tax_rev_top_tax(return_micro=True, tau=tau, zeta=zeta)
        if T < 0.0:
            return 1e6

        # 10.4 objective: minus SWF_top
        return -self.SWF_top(tau=tau, zeta=zeta)

                
    # 11. maximize SWF over (tau, zeta, omega, kappa)
    def max_policies(self, x0=None, verbose=True):

        par = self.par
        sol = self.sol

        # 11.1 initial guess
        if x0 is None:
            # x0 = [tau, zeta, omega, kappa]
            x0 = np.array([0.1, 0.0, 0.1, 9.0])

        # 11.2 run optimizer
        res = optimize.minimize(
            fun=self.obj_policies,
            x0=x0,
            method="Nelder-Mead",
        )

        # 11.3 unpack results
        tau_star, zeta_star, omega_star, kappa_star = res.x

        sol.tau_star   = tau_star
        sol.zeta_star  = zeta_star
        sol.omega_star = omega_star
        sol.kappa_star = kappa_star
        sol.SWF_top_star = -res.fun 

        # 11.4 baseline SWF without top tax at same (tau*, zeta*)
        omega_old, kappa_old = par.omega, par.kappa

        par.omega = 0.0  
        par.kappa = kappa_star
        swf_no_top = self.SWF(tau=tau_star, zeta=zeta_star)

        par.omega = omega_old
        par.kappa = kappa_old

        if verbose:
            print("\n=== Numerical solution for optimal tax system (tau, zeta, omega, kappa) ===\n")
            print(f"Optimal linear tax (tau*)     : {tau_star:.4f}")
            print(f"Optimal transfer (zeta*)      : {zeta_star:.4f}")
            print(f"Optimal top tax (omega*)      : {omega_star:.4f}")
            print(f"Optimal threshold (kappa*)    : {kappa_star:.4f}")
            print("------------------------------------------------")
            print(f"SWF without top tax           : {swf_no_top:.6f}")
            print(f"Max SWF with top tax          : {sol.SWF_top_star:.6f}")
            print(f"Gain (top - base)             : {sol.SWF_top_star - swf_no_top:.6f}")
            print(f"Converged?                    : {res.success}\n")

        return res
            
        
        
        
        
        
    
        
        
        
        
        
        