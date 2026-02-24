import numpy as np

###########################################################
# 1. Demand- and supply elasticities
###########################################################

# ========== ========== ========== ========== ========== 
# 1. elasticities (q,K) 
def dem_sup_elas(m, tau):
    
    # 1. get
    ss = m.solve_steady_state(tau=tau)
    vals = m.static_block_sigmoid(K=ss['K'], q=ss['q'], tau=ss['tau'])
    
    # 2. compute via matrix
    A_theta = np.array([
    [ 
            vals['sK']/(1-vals['sK']) * (m.alphaK-1) + (m.betaK-1)  ,
            vals['sL']/(1-vals['sL']) * m.alphaL + m.betaL          , 
            1.0
    ]   ,
    [
            vals['sK']/(1-vals['sK']) * m.alphaK + m.betaK          ,
            vals['sL']/(1-vals['sL']) * (m.alphaL-(1+m.phi)/m.phi)
            + m.betaL - (1+m.phi)/m.phi                             ,
            1.0
    ]   ,
    [
            m.theta*m.betaK, m.theta*m.betaL, 1.0
    ]
    ])
    
    A0 = A_theta.copy()
    A0[2, :] = np.array([0.0, 0.0, 1.0])

    b = np.array([0.0, 0.0, 1.0])
    
    # 3. invert
    x_theta = np.linalg.solve(A_theta, b)
    x0      = np.linalg.solve(A0, b)
    
    # 4. compute demand elasticity
    demand_vec = np.array([
        vals['sK']/(1-vals['sK'])                                   , 
        -m.alphaL/(1-m.alphaK) * vals['sL']/(1-vals['sL'])          ,
        0.0
    ])
    epsD = - (-1/(1-m.alphaK) + demand_vec @ x0)

    # 5. compute supply elasticity
    supply_vec = np.array([ m.betaK, m.betaL, 0.0 ])
    epsS_LR = supply_vec @ x0
    epsS_SR = m.delta * (1 - m.theta) * (supply_vec @ x_theta)
    
    # 6. output dict
    out = {
        'epsD': epsD,
        'epsS_LR': epsS_LR,
        'epsS_SR': epsS_SR
    }
    
    return out
    
    
# ========== ========== ========== ========== ========== 
# 2. Q/P tax elasticities
def qp_tax_elas(m, out):
    
    # 1. get
    epsD = out['epsD']
    epsS_LR = out['epsS_LR']
    
    # 2. compute effects
    price_elas  =  1 / (1-m.alphaK) * 1 / (epsS_LR + epsD)
    quant_elas  =  1 / (1-m.alphaK) * epsS_LR / (epsS_LR + epsD)
    wealth_elas = price_elas + quant_elas
    
    # 3. out dict
    out = {
        'price_elas': price_elas,
        'quant_elas': quant_elas,
        'wealth_elas': wealth_elas
    }
    
    return out


# ========== ========== ========== ========== ========== 
# 3. wage/rent tax elasticities
def wr_tax_elas(m, elas_out, phi=None, epsD=None, epsS_LR=None):
    
    # 1. get elasticities
    _epsD    = epsD    if epsD    is not None else elas_out['epsD']
    _epsS_LR = epsS_LR if epsS_LR is not None else elas_out['epsS_LR']
    
    # optionally override labor supply elasticity
    if phi is not None:
        m.phi = phi

    # 2. compute effects
    w_C_elas = m.alphaK * _epsS_LR        / ((1 - m.alphaK) * (_epsS_LR + _epsD))
    w_I_elas = (1 + m.betaK * _epsS_LR)   / ((1 - m.alphaK) * (_epsS_LR + _epsD))
    r_K_elas = 1                          / ((1 - m.alphaK) * (_epsS_LR + _epsD))
    w_I_breakdown = (m.betaK * _epsS_LR)   / ((1 - m.alphaK) * (_epsS_LR + _epsD))

    return {
        'w_C_elas': w_C_elas,
        'w_I_elas': w_I_elas,
        'r_K_elas': r_K_elas,
        'w_I_breakdown':w_I_breakdown,
    }
    

# ========== ========== ========== ========== ========== 
# 4. welfare shares

def welfare_incidence(sim):

    # 1. tax incidence 
    consump_w = sim["table"]["pv_wg_C"] / sim["table"]["pv_WG"]
    investm_w = sim["table"]["pv_wg_I"] / sim["table"]["pv_WG"]
    capital_o = sim["table"]["pv_wg_K"] / sim["table"]["pv_WG"]
        
    return {
        'consump_w': consump_w,
        'investm_w': investm_w,
        'capital_o': capital_o,
    }
    
    