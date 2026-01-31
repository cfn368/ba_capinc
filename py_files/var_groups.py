
model_var = {
    'K':        r'$K$',
    'q':        r'$q$',
    'pI':       r'$p_I$',
    'I':        r'$I$',
    'C':        r'$C$',
    'rC_gross': r'$r_K$',
    'sK':       r'$s_K$',
    's1':       r'$s_{1L}$',
    's2':       r'$s_{2L}$',
    'L1C':      r'$L_{1C}$',
    'L2C':      r'$L_{2C}$',
    'L1I':      r'$L_{1I}$',
    'L2I':      r'$L_{2I}$',
    'tau':      r'$\tau$',
    'Y':        r'$Y$',
    'w1C':      r'$w_{1C}$',
    'w2C':      r'$w_{2C}$',
    'w1I':      r'$w_{1I}$',
    'w2I':      r'$w_{2I}$',
}

def panels(gamma): 
    return [
    # ("Y", r"$Y$"),
    # ("C", r"$C$"),
    # ('pII', r"$p_II$"),
    # ("wC", r"$w_C$"),
    # ("wI", r"$w_I$"),
    ("LS_C", r"$LS_C$"),
    ("LS_I", r"$LS_I$"),
    ("LS", r"$LS$"),
    ("LS_gamma", rf"Adj. $LS,\; (\gamma={gamma})$"),
]