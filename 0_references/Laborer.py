from types import SimpleNamespace


import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
from scipy import optimize

class labourer_class:

    # 1. initiate
    def __init__(self,par=None):
        
        # 1.1 setup 
        self.setup_worker()
        
        # 1.2 update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v
           
                
    # 2. define worker parameters
    def setup_worker(self):
        
        # 2.1 lists for parameters and solution
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        
        # 2.2 calibration params
        par.ell_max = 16
        par.w       = 1
        par.tau     = 0.5
        par.zeta    = 0.1
        par.epsilon = 1
        par.nu      = 0.015
        par.p       = 1
        par.kappa   = 9
        par.omega   = 0.2
        
        
    # 3. utility func of ell and p
    def utility(self,ell):
        
        # 3.1 get parameters
        par = self.par
        
        # 3.2 define c(ell)
        c = (1-par.tau) * par.w * par.p * ell - par.zeta 
        
        # 3.3 define u(ell)
        U = np.log(c) - par.nu * ell**(1+par.epsilon) / (1+par.epsilon)
        
        return U
    
    
    # 4. define FOC
    def FOC(self,ell):
        
        # 4.1 get params
        par = self.par
        c = (1-par.tau) * par.w * par.p * ell - par.zeta
        
        # 4.2 define phi(ell)
        varphi = ((1-par.tau) * par.w * par.p) / c - par.nu * ell**par.epsilon
        
        return varphi
    
    
    # 5. numerical solver of V(p)
    def numerical_optimization(self,verbose=True):
        
        par         = self.par
        sol         = self.sol
        
        # 5.1 define objective
        def obj(ell): return -self.utility(ell)
        
        # 5.2 set up constraint
        ell_min = max(par.zeta / ((1 - par.tau) * par.w * par.p), 0)
        
        # 5.3 set up solver
        res = (
            optimize.minimize_scalar(
                obj, 
                bounds=(ell_min, par.ell_max), 
                method='bounded')
            )
        
        # 5.4 solution
        sol.ell_star = res.x
        sol.U_star   = -res.fun
        
        # 5.5 print
        if verbose==True:
            print("\n=== Numerical solution for optimal labour supply ===\n")
            print(f"Price (p)        : {par.p:.3f}")
            print(f"Wage (w)         : {par.w:.3f}")
            print(f"Tax rate (tau)   : {par.tau:.3f}")
            print(f"Transfer (zeta)  : {par.zeta:.3f}")
            print(f"ell bounds       : [{ell_min:.3f}, {par.ell_max:.3f}]")
            print("----------------------------------------------------")
            print(f"Optimal hours ell* : {sol.ell_star:4f}")
            print(f"Utility U(ell*)    : {sol.U_star:4f}\n")
        
    
    # 6. root-finder for FOC
    def root_finder(self, verbose=True):
        
        par         = self.par
        sol         = self.sol
        
        # 6.1 ensure positive consumption
        ell_min = max(par.zeta / ((1 - par.tau) * par.w * par.p), 0)
        
        # 6.2 set up solver
        res = optimize.newton(
            func=self.FOC, 
            x0=8,
            tol=1e-10,
            maxiter=1000,
        )
    
        # 6.3 impose bounds (hard coded)
        sol.ell_star_root = res
        if sol.ell_star_root < ell_min      : sol.ell_star_root = ell_min
        if sol.ell_star_root > par.ell_max  : sol.ell_star_root = par.ell_max

        # 6.4 solution
        sol.U_star_root   = self.utility(sol.ell_star_root)
        sol.FOC_at_root   = self.FOC(sol.ell_star_root)
        
        # 6.5 print 
        if verbose==True:
            print("\n=== Root-finder solution for optimal labour supply ===\n")
            print(f"Price (p)        : {par.p:.3f}")
            print(f"Wage (w)         : {par.w:.3f}")
            print(f"Tax rate (tau)   : {par.tau:.3f}")
            print(f"Transfer (zeta)  : {par.zeta:.3f}")
            ell_min = par.zeta / ((1 - par.tau) * par.w * par.p)
            print(f"ell bounds       : [{ell_min:.3f}, {par.ell_max:.3f}]")
            print("----------------------------------------------------")
            print(f"Optimal hours ell* : {sol.ell_star_root:4f}")
            print(f"Utility U(ell*)    : {sol.U_star_root:4f}\n")
            
            
    # With TOP-TAX #####################################################################
    
    # I redo many of the above, with the new expression for c
    
    
    # 7. pre tax income
    def gross_income(self, ell):
        par = self.par
        return par.w * par.p * ell
    
    
    # 8. post-tax income
    def income_top(self, ell):
        par = self.par
        y_gross = self.gross_income(ell)
        top_part = par.omega * np.maximum(y_gross - par.kappa, 0.0)
        c = (1 - par.tau) * y_gross - top_part - par.zeta
        return c
    
    
    # 9. utility top income group
    def utility_top(self, ell):
        c = self.income_top(ell)
        if np.any(c <= 0):
            return -1e10 
        par = self.par
        return np.log(c) - par.nu * ell**(1 + par.epsilon) / (1 + par.epsilon)
    
    # 10. above treshold
    def FOC_above(self, ell):
        par = self.par
        
        # 10.1 specific c
        c = ((1 - par.tau - par.omega) * par.w * par.p * ell
             + par.omega * par.kappa - par.zeta)
        
        # 10.2 return FOC
        return (1 - par.tau - par.omega) * par.w * par.p / c - par.nu * ell**par.epsilon
    
    
    # 11 optimize step-wise problem (root)
    def solve_top_tax_root(self, verbose=True):
        par = self.par
        sol = self.sol

        # 11.1 find kink
        ell_k = par.kappa / (par.w * par.p)
        ell_k = np.clip(ell_k, 0, par.ell_max)
        par.ell_k = ell_k

        # 11.2 set tolerences
        eps = 1e-6
        ell_lo = 0.5

        # Case 'b'
        ell_hi_b = min(ell_k - eps, par.ell_max)
        ell_b, U_b = None, -np.inf
        
        if ell_hi_b > ell_lo:
            try:
                
                # b.1 solve FOC
                root_b = optimize.root_scalar(
                    self.FOC,
                    bracket=[ell_lo, ell_hi_b],
                    method="brentq",
                )
                
                # b.2 compute utility
                if root_b.converged:
                    ell_b = root_b.root
                    U_b = self.utility(ell_b)
            except ValueError:
                pass

        # case 'k'
        ell_k_used = ell_k
        U_k = self.utility(ell_k_used)

        # case 'a'
        ell_lo_a = max(ell_k + eps, ell_lo)
        ell_hi_a = par.ell_max
        ell_a, U_a = None, -np.inf
        
        if ell_hi_a > ell_lo_a:
            try:
                # a.1 solve FOC
                root_a = optimize.root_scalar(
                    self.FOC_above,
                    bracket=[ell_lo_a, ell_hi_a],
                    method="brentq",
                )
                
                # a.2 compute utility
                if root_a.converged:
                    ell_a = root_a.root
                    U_a = self.utility_top(ell_a)
            except ValueError:
                pass

        # 11.3 pick best
        U_list   = [U_b, U_k, U_a]
        ell_list = [ell_b, ell_k_used, ell_a]
        label    = ['b', 'k', 'a']

        i = int(np.argmax(U_list))
        ell_star = ell_list[i]
        U_star   = U_list[i]
        t_star   = label[i]

        # 11.4 results and print
        sol.ell_b_top  = ell_b
        sol.ell_k_top  = ell_k_used
        sol.ell_a_top  = ell_a
        sol.ell_star_top = ell_star
        sol.U_star_top   = U_star
        sol.type_star_top = t_star
        
        if t_star == 'b':
            FOC_at_root = self.FOC(ell_b)
        elif t_star == 'a':
            FOC_at_root = self.FOC_above(ell_a)
        else: 
            FOC_at_root = self.FOC(ell_k_used)

        sol.FOC_at_root_top_tax = FOC_at_root

        if verbose:
            print("\n=== ROOT solution for labour supply with top-tax ===\n")
            print(f"Price (p)          : {par.p:.3f}")
            print(f"Wage (w)           : {par.w:.3f}")
            print(f"Tax rate (tau)     : {par.tau:.3f}")
            print(f"Transfer (zeta)    : {par.zeta:.3f}")
            print(f"Top tax (omega)    : {par.omega:.3f}")
            print(f"Kink (kappa)       : {par.kappa:.3f}")
            print(f"ell kink (ell_k)   : {par.ell_k:.3f}")
            print("----------------------------------------------------")
            print(f"Region type        : {t_star}")
            print(f"Optimal hours ell* : {ell_star: .4f}")
            print(f"Utility U(ell*)    : {U_star: .4f}\n")

        return ell_star, t_star, U_star, FOC_at_root

    
    # 12. optimize step-wise problem (num)
    def solve_top_tax_num(self, verbose=True):
        par = self.par
        sol = self.sol

        # 12.1 kink
        ell_k = par.kappa / (par.w * par.p)
        ell_k = np.clip(ell_k, 0, par.ell_max)
        par.ell_k = ell_k

        # 12.2 feasible interval (positive consumption under top-tax)
        ell_min = 0.5
        ell_max = par.ell_max

        # 12.3 run optimizer
        def obj(ell):
            return -self.utility_top(ell)

        res = optimize.minimize_scalar(
            obj,
            bounds=(ell_min, ell_max),
            method="bounded",
        )

        # 12.4 get result
        ell_star = res.x
        U_star   = self.utility_top(ell_star)

        # 12.5 classify region ex post
        if ell_star < ell_k - 1e-6:
            t_star = 'b'
        elif ell_star > ell_k + 1e-6:
            t_star = 'a'
        else:
            t_star = 'k'

        # 12.6 save and print
        sol.ell_star_top_num = ell_star
        sol.U_star_top_num   = U_star
        sol.type_star_top_num = t_star

        if verbose:
            print("\n=== NUMERICAL solution for labour supply with top-tax ===\n")
            print(f"Price (p)          : {par.p:.3f}")
            print(f"Wage (w)           : {par.w:.3f}")
            print(f"Tax rate (tau)     : {par.tau:.3f}")
            print(f"Fixed cost (zeta)  : {par.zeta:.3f}")
            print(f"Top tax (omega)    : {par.omega:.3f}")
            print(f"Kink (kappa)       : {par.kappa:.3f}")
            print(f"ell kink (ell_k)   : {par.ell_k:.3f}")
            print("----------------------------------------------------")
            print(f"Region type        : {t_star}")
            print(f"Optimal hours ell* : {ell_star: .4f}")
            print(f"Utility U(ell*)    : {U_star: .4f}\n")
        return ell_star, t_star
        
        
            
    
