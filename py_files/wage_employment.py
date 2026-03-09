"""
Sectoral wages & employment: continuous-weight evidence for φ
=============================================================

Computes w_I/w_C and L_I/L_C using continuous Leontief-derived weights
from the sectoral_labor_share module.  The combined figure provides
empirical motivation for declining specialised labour supply elasticity.

Data sources:
    NABP36  — D.1 compensation of employees by industry
    NABB36  — Hours worked / employees by industry
    NAIO1F  — Input-output tables (via leontief_analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import py_files.var_groups as var_groups
from py_files.inverse_leontif import compute_leontief_for_year
from py_files.LS_aggregator import sectoral_labor_shares, consolidated_labor_shares

from dstapi import DstApi


# ========== ========== ========== ========== ========== ==========
#  helpers
# ========== ========== ========== ========== ========== ==========
def _code_token(x):
    s = pd.Series(x, dtype="string")
    return s.str.extract(r"^\s*([^ ]+)", expand=False)


# ========== ========== ========== ========== ========== ==========
# 1. Fetch compensation by industry (all years)
# ========== ========== ========== ========== ========== ==========
def fetch_compensation():
    """
    D.1 compensation of employees from NABP36, current prices.
    Returns long DataFrame: year, branche_code, compensation
    """
    NABP36 = DstApi('NABP36')

    params = {
        'table': 'NABP36',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED', 'values': ['V']},
            {'code': 'TRANSAKT',  'values': ['D1D']},
            {'code': 'Tid',       'values': ['*']},
            {'code': 'BRANCHE',   'values': ['*']},
        ]
    }

    df = NABP36.get_data(params=params)
    df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')
    df = df[~df['BRANCHE'].eq('Total')].copy()
    df['branche_code'] = _code_token(df['BRANCHE'])
    df = df[~df['branche_code'].eq('Of')].copy()
    df['year'] = df['TID'].astype(int)

    return df[['year', 'branche_code', 'INDHOLD']].rename(
        columns={'INDHOLD': 'compensation'}
    )


# ========== ========== ========== ========== ========== ==========
# 2. Fetch hours worked by industry (all years)
# ========== ========== ========== ========== ========== ==========
def fetch_hours():
    """
    Hours worked for employees from NABB36 (1,000 hours).
    Returns long DataFrame: year, branche_code, hours
    """
    NABB36 = DstApi('NABB36')

    params = {
        'table': 'NABB36',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'SOCIO',   'values': ['SALH_DC']},
            {'code': 'BRANCHE', 'values': ['*']},
            {'code': 'Tid',     'values': ['*']},
        ]
    }

    df = NABB36.get_data(params=params)
    df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')
    df = df[~df['BRANCHE'].eq('Total')].copy()
    df['branche_code'] = _code_token(df['BRANCHE'])
    df = df[~df['branche_code'].eq('Of')].copy()
    df['year'] = df['TID'].astype(int)

    return df[['year', 'branche_code', 'INDHOLD']].rename(
        columns={'INDHOLD': 'hours'}
    )


# ========== ========== ========== ========== ========== ==========
# 3. Fetch number of employees by industry (all years)
# ========== ========== ========== ========== ========== ==========
def fetch_employees():
    """
    Number of employees from NABB36.
    Returns long DataFrame: year, branche_code, employees
    """
    NABB36 = DstApi('NABB36')

    params = {
        'table': 'NABB36',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'SOCIO',   'values': ['SALM_DC']},
            {'code': 'BRANCHE', 'values': ['*']},
            {'code': 'Tid',     'values': ['*']},
        ]
    }

    df = NABB36.get_data(params=params)
    df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')
    df = df[~df['BRANCHE'].eq('Total')].copy()
    df['branche_code'] = _code_token(df['BRANCHE'])
    df = df[~df['branche_code'].eq('Of')].copy()
    df['year'] = df['TID'].astype(int)

    return df[['year', 'branche_code', 'INDHOLD']].rename(
        columns={'INDHOLD': 'employees'}
    )


# ========== ========== ========== ========== ========== ==========
# 4. Get continuous weights for a given year
# ========== ========== ========== ========== ========== ==========
def _get_parent_weights(year, kappa=0.6):
    """
    Run Leontief for `year` and return continuous weights
    aggregated to the 36a2 parent level.

    Returns dict with:
        'w_C' : Series indexed by parent code
        'w_I' : Series indexed by parent code
    """
    from py_files.LS_aggregator import (
        fetch_industry_labor_shares,
        consolidated_labor_shares,
        sectoral_labor_shares,
    )

    yr = compute_leontief_for_year(year)

    # need LS data to run consolidated_labor_shares (even though
    # we only want the weights here, the function flow requires it)
    df_ls = fetch_industry_labor_shares()
    df_ls_yr = df_ls[df_ls['year'] == year]

    ls_cons = consolidated_labor_shares(yr, df_ls_yr)
    res = sectoral_labor_shares(yr, ls_cons, kappa=kappa)

    # aggregate sub-industry weights to parent level
    parent_map = var_groups.sub_to_parent
    w_C_sub = res['weights_C']
    w_I_sub = res['weights_I']

    w_df = pd.DataFrame({
        'parent': [parent_map.get(s, s) for s in w_C_sub.index],
        'w_C': w_C_sub.values,
        'w_I': w_I_sub.values,
    })

    w_parent = w_df.groupby('parent')[['w_C', 'w_I']].sum()
    return w_parent


# ========== ========== ========== ========== ========== ==========
# 5. Compute w_I/w_C and L_I/L_C timeseries
# ========== ========== ========== ========== ========== ==========
def compute_wage_employment_timeseries(years, kappa=0.6):
    """
    Compute sectoral wage ratio w_I/w_C and employment ratio L_I/L_C
    using continuous Leontief weights, computed fresh for every year.

    Parameters
    ----------
    years : iterable
        Years for the output timeseries.
    kappa : float
        Org services capitalisation rate.

    Returns
    -------
    DataFrame indexed by year with columns:
        w_I, w_C, w_ratio, L_I, L_C, L_ratio
    """
    years = sorted(years)

    # --- 1. fetch all industry-level data once ---
    print("Fetching compensation (NABP36) ...")
    df_comp = fetch_compensation()
    print("Fetching hours (NABB36) ...")
    df_hours = fetch_hours()
    print("Fetching employees (NABB36) ...")
    df_empl = fetch_employees()

    rows = []
    for year in years:
        try:
            # --- Leontief weights for this year ---
            w_parent = _get_parent_weights(year, kappa=kappa)

            # --- industry data for this year ---
            comp_yr  = df_comp[df_comp['year'] == year].set_index('branche_code')['compensation']
            hours_yr = df_hours[df_hours['year'] == year].set_index('branche_code')['hours']
            empl_yr  = df_empl[df_empl['year'] == year].set_index('branche_code')['employees']

            if comp_yr.empty or hours_yr.empty or empl_yr.empty:
                print(f"  {year}: missing NABP/NABB data, skipping")
                continue

            # --- wages: align on common codes with comp & hours ---
            common_w = (w_parent.index
                        .intersection(comp_yr.index)
                        .intersection(hours_yr.index))

            if len(common_w) < 10:
                print(f"  {year}: only {len(common_w)} industries matched, skipping")
                continue

            w_C = w_parent.loc[common_w, 'w_C']
            w_I = w_parent.loc[common_w, 'w_I']
            comp = comp_yr[common_w]
            hrs  = hours_yr[common_w]

            # sectoral wage = total sector compensation / total sector hours
            wage_C = (w_C * comp).sum() / (w_C * hrs).sum()
            wage_I = (w_I * comp).sum() / (w_I * hrs).sum()

            # --- employment: align on common codes with employees ---
            common_e = w_parent.index.intersection(empl_yr.index)
            w_C_e = w_parent.loc[common_e, 'w_C']
            w_I_e = w_parent.loc[common_e, 'w_I']
            empl  = empl_yr[common_e]

            L_total = empl.sum()
            L_I = (w_I_e * empl).sum() / L_total
            L_C = (w_C_e * empl).sum() / L_total

            rows.append({
                'year':    year,
                'w_C':     wage_C,
                'w_I':     wage_I,
                'w_ratio': wage_I / wage_C if wage_C > 0 else np.nan,
                'L_I':     L_I,
                'L_C':     L_C,
                'L_ratio': L_I / L_C if L_C > 0 else np.nan,
            })
            print(f"  {year}: w_I/w_C={wage_I/wage_C:.3f}  L_I/L_C={L_I/L_C:.3f}")

        except Exception as e:
            print(f"  {year}: error — {e}")
            continue

    df = pd.DataFrame(rows).set_index('year')
    print("Done.")
    return df


# ========== ========== ========== ========== ========== ==========
# 6. Plot: w_I/w_C and L_I/L_C on dual axes
# ========== ========== ========== ========== ========== ==========
def plot_wage_employment(df, save_path='0_output/wage_employment.png'):
    """
    Single figure, dual y-axes:
      Left:  w_I / w_C  (wage ratio)
      Right: L_I / L_C  (employment ratio)

    Parameters
    ----------
    df : DataFrame from compute_wage_employment_timeseries()
    save_path : str or None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax_r = ax.twinx()

    # --- left axis: wage ratio ---
    l1, = ax.plot(
        df.index, df['w_ratio'],
        color='crimson', lw=2, ls='-',
    )
    ax.axhline(1, color='0.2', linewidth=1, ls='--')
    ax.set_ylabel(r'$w_I\,/\,w_C$')
    ax.set_xlabel('Year')

    # --- right axis: employment ratio ---
    l2, = ax_r.plot(
        df.index, df['L_ratio'],
        color='#1F2A44', lw=2, ls='--',
    )
    ax_r.set_ylabel(r'$L_I\,/\,L_C$')

    # --- formatting ---
    w_vals = df['w_ratio'].dropna()
    L_vals = df['L_ratio'].dropna()

    w_dev = max(w_vals.max() - 1, 1 - w_vals.min()) * 1.15   # 15% padding
    L_dev = max(L_vals.max() - 1, 1 - L_vals.min()) * 1.15

    ax.set_ylim(1 - w_dev, 1 + w_dev)
    ax_r.set_ylim(1 - L_dev, 1 + L_dev)
    ax.grid(linewidth=0.6, alpha=0.35)
    
    ax.set_xlim(df.index.min(), df.index.max())
    ax.grid(linewidth=0.6, alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    ax.legend(
        [l1, l2],
        [r'Wage ratio $w_I/w_C$', r'Employment ratio $L_I/L_C$'],
        loc='upper right'
    )

    # fig.suptitle(
    #     'Sectoral wage and employment ratios (continuous weights)',
    #     y=0.95
    # )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return fig, ax