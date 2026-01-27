import matplotlib as mpl

def set_aej():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 15,

        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "legend.frameon": False,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # legend
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.borderaxespad": 0.4,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,

        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })