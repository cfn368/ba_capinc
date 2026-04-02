"""
Sectoral wages & employment: continuous-weight evidence for φ
=============================================================

Computes w_I/w_C and L_I/L_C using continuous direct IO weights
from the sectoral_labor_share module.  The combined figure provides
empirical motivation for declining specialised labour supply elasticity.

Data sources:
    NABP36  — D.1 compensation of employees by industry
    NABB36  — Hours worked / employees by industry
    NAIO1F  — Input-output tables (via direct_NX)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import py_files.var_groups as var_groups
from py_files.direct_NX import compute_direct_for_year, load_or_compute_year, CACHE_DIR
from py_files.LS_aggregator import (
    sectoral_labor_shares, consolidated_labor_shares,
    load_or_fetch_industry_labor_shares,
)

from dstapi import DstApi


# ==================== ==================== ==================== ====================
# 0. helpers

def _code_token(x):
    s = pd.Series(x, dtype="string")
    return s.str.extract(r"^\s*([^ ]+)", expand=False)


# ==================== ==================== ==================== ====================
# 1. Fetch compensation by industry (all years)

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


# ==================== ==================== ==================== ====================
# 2. Fetch hours worked by industry (all years)

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


# ==================== ==================== ==================== ====================
# 3. Fetch number of employees by industry (all years)

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


# ==================== ==================== ==================== ====================
# 3b. cached NABB/NABP fetchers

def load_or_fetch_compensation(cache_dir=CACHE_DIR, force=False):
    path = os.path.join(cache_dir, "nabb_compensation.parquet")
    if not force and os.path.exists(path):
        print("  Loading compensation from cache …")
        return pd.read_parquet(path)
    print("Fetching compensation (NABP36) …")
    df = fetch_compensation()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_or_fetch_hours(cache_dir=CACHE_DIR, force=False):
    path = os.path.join(cache_dir, "nabb_hours.parquet")
    if not force and os.path.exists(path):
        print("  Loading hours from cache …")
        return pd.read_parquet(path)
    print("Fetching hours (NABB36) …")
    df = fetch_hours()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_or_fetch_employees(cache_dir=CACHE_DIR, force=False):
    path = os.path.join(cache_dir, "nabb_employees.parquet")
    if not force and os.path.exists(path):
        print("  Loading employees from cache …")
        return pd.read_parquet(path)
    print("Fetching employees (NABB36) …")
    df = fetch_employees()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ==================== ==================== ==================== ====================
# 4. Get continuous weights for a given year

def _get_parent_weights(year, kappa=0.6, df_ls_all=None, use_cache=False,
                        cache_dir=CACHE_DIR):
    """
    Run direct IO analysis for `year` and return continuous weights
    aggregated to the 36a2 parent level.

    Returns dict with:
        'w_C' : Series indexed by parent code
        'w_I' : Series indexed by parent code
    """
    yr = load_or_compute_year(year, cache_dir=cache_dir) if use_cache \
         else compute_direct_for_year(year)

    if df_ls_all is None:
        from py_files.LS_aggregator import fetch_industry_labor_shares
        df_ls_all = fetch_industry_labor_shares()

    df_ls_yr = df_ls_all[df_ls_all['year'] == year]

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


# ==================== ==================== ==================== ====================
# 5. Compute w_I/w_C and L_I/L_C timeseries

def compute_wage_employment_timeseries(years, kappa=0.6, use_cache=False,
                                        cache_dir=CACHE_DIR):
    """
    Compute sectoral wage ratio w_I/w_C and employment ratio L_I/L_C
    using continuous Leontief weights.

    Parameters
    ----------
    years     : iterable
    kappa     : float  — org services capitalisation rate
    use_cache : bool   — use cached IO tables and NABP/NABB data
    cache_dir : str    — directory for cache files

    Returns
    -------
    DataFrame indexed by year with columns:
        w_I, w_C, w_ratio, L_I, L_C, L_ratio
    """
    years = sorted(years)

    # 1. fetch all industry-level data once (optionally cached)
    if use_cache:
        df_comp  = load_or_fetch_compensation(cache_dir=cache_dir)
        df_hours = load_or_fetch_hours(cache_dir=cache_dir)
        df_empl  = load_or_fetch_employees(cache_dir=cache_dir)
        df_ls_all = load_or_fetch_industry_labor_shares(cache_dir=cache_dir)
    else:
        print("Fetching compensation (NABP36) ...")
        df_comp = fetch_compensation()
        print("Fetching hours (NABB36) ...")
        df_hours = fetch_hours()
        print("Fetching employees (NABB36) ...")
        df_empl = fetch_employees()
        df_ls_all = None

    rows = []
    for year in years:
        try:
            # 2. Leontief weights for this year
            w_parent = _get_parent_weights(year, kappa=kappa,
                                           df_ls_all=df_ls_all,
                                           use_cache=use_cache,
                                           cache_dir=cache_dir)

            # 3. industry data for this year
            comp_yr  = df_comp[df_comp['year'] == year].set_index('branche_code')['compensation']
            hours_yr = df_hours[df_hours['year'] == year].set_index('branche_code')['hours']
            empl_yr  = df_empl[df_empl['year'] == year].set_index('branche_code')['employees']

            if comp_yr.empty or hours_yr.empty or empl_yr.empty:
                print(f"  {year}: missing NABP/NABB data, skipping")
                continue

            # 4. wages: align on common codes with comp & hours
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

            # 5. employment: align on common codes with employees
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


# ==================== ==================== ==================== ====================
# 6. Plot: w_I/w_C and L_I/L_C on dual axes

def plot_wage_employment(df, save_path='0_output/wage_employment.png'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # 1. wage ratio
    l1, = ax.plot(
        df.index, df['w_ratio'],
        color='crimson', lw=2, ls='-',
    )

    # 2. employment ratio (same axis)
    l2, = ax.plot(
        df.index, df['L_ratio'],
        color='#1F2A44', lw=2, ls='--',
    )

    ax.axhline(1, color='0.2', linewidth=1, ls='--')
    ax.set_ylabel(r'Ratio')
    ax.set_xlabel('Year')

    ax.grid(linewidth=0.6, alpha=0.35)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    ax.legend(
        [l1, l2],
        [r'Wage ratio $w_I/w_C$', r'Employment ratio $L_I/L_C$'],
        loc='lower left'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return fig, ax


# ==================== ==================== ==================== ====================
# 7. Cached wrapper

def load_or_compute_we_timeseries(years, kappa=0.6,
                                   cache_path=None, cache_dir=CACHE_DIR, force=False):
    """
    Return compute_wage_employment_timeseries(...), reading from parquet cache
    if available.

    Parameters
    ----------
    years      : iterable of ints
    kappa      : passed through
    cache_path : explicit parquet path; defaults to
                 <cache_dir>/we_timeseries_<start>_<end>.parquet
    cache_dir  : directory for all cache files
    force      : if True, ignore existing cache and recompute
    """
    years = sorted(years)
    if cache_path is None:
        fname = f"we_timeseries_{years[0]}_{years[-1]}.parquet"
        cache_path = os.path.join(cache_dir, fname)

    if not force and os.path.exists(cache_path):
        print(f"Loading wage-employment timeseries from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    df = compute_wage_employment_timeseries(years, kappa=kappa,
                                             use_cache=True, cache_dir=cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"  Saved to {cache_path}")
    return df
