import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import py_files.var_groups as var_groups

from py_files.direct_NX import (compute_direct_for_year, load_or_compute_year,
                                 CACHE_DIR)
from dstapi import DstApi


# ==================== ==================== ==================== ====================
# 1. Fetch direct labor shares from NABP36 (industry × year)

def fetch_industry_labor_shares():
    """
    Fetch D.1 (compensation) and B.1g (GVA) by industry and year
    from NABP36. Returns a long DataFrame with columns:
        year, branche_code, e_comp, GVA
    """
    NABP36 = DstApi('NABP36')

    params = {
        'table':  'NABP36',
        'format': 'BULK',
        'lang':   'en',
        'variables': [
            {'code': 'PRISENHED', 'values': ['V']},       # current prices
            {'code': 'Tid',       'values': ['*']},
            {'code': 'BRANCHE',   'values': ['*']},
            {'code': 'TRANSAKT',  'values': ['B1GD', 'D1D']},
        ]
    }

    df = NABP36.get_data(params=params)
    df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

    df = df.pivot_table(
        index=['TID', 'BRANCHE'],
        columns='TRANSAKT',
        values='INDHOLD'
    ).reset_index()

    df = df.rename(columns={
        'B.1g Gross value added':            'GVA',
        'D.1 Compensation of employees':     'e_comp',
    })

    # drop aggregate row
    df = df[~df['BRANCHE'].eq('Total')].copy()

    # extract leading industry code
    df['branche_code'] = (
        df['BRANCHE'].astype(str)
        .str.extract(r'^\s*([^ ]+)', expand=False)
    )
    df['year'] = df['TID'].astype(int)

    return df[['year', 'branche_code', 'e_comp', 'GVA']].copy()


# ==================== ==================== ==================== ====================
# 2. Labour share per industry (direct or Leontief-consolidated)

def consolidated_labor_shares(year_result, df_ls_year, use_leontief=False):
    """
    Compute the labour share for each sub-industry.

    Direct (use_leontief=False):
        Returns CE_j / GVA_j at parent level, mapped to sub-industries.

    Leontief (use_leontief=True):
        Computes ℓ_j = CE_j / X_j (compensation per unit of gross output),
        then returns ℓ' · (I − A)^{-1}, following GG-B.  This gives the
        total labour compensation embodied per unit of each industry's
        gross output, tracing both direct and upstream contributions.

    Parameters
    ----------
    year_result : dict
        Output of compute_direct_for_year(). Uses 'X' (gross output) and
        'Z' (intermediate use matrix).
    df_ls_year : DataFrame
        Labour-share data for a *single year*, columns:
        'branche_code' (39-parent code), 'e_comp', 'GVA'.
    use_leontief : bool

    Returns
    -------
    Series indexed by sub-industry code (ratio, not %).
    """
    subs       = year_result['X'].index
    X          = year_result['X']
    Z          = year_result['Z']
    parent_map = var_groups.sub_to_parent
    parent_ls  = df_ls_year.set_index('branche_code')

    if use_leontief:
        # 1. ℓ_j = CE_j / X_j — CE is at parent level; all subs within a parent
        # share the same ℓ = CE_parent / X_parent_total (proportional allocation
        # cancels out when dividing by sub-industry X).
        x_parent_total = {}
        for s in subs:
            p = parent_map.get(s, s)
            if p not in x_parent_total:
                parent_subs = [ss for ss in subs if parent_map.get(ss, ss) == p]
                x_parent_total[p] = X[parent_subs].sum()

        ce_series = parent_ls['e_comp']
        ell = pd.Series(
            [ce_series.get(parent_map.get(s, s), 0.0) /
             x_parent_total.get(parent_map.get(s, s), np.nan)
             for s in subs],
            index=subs
        ).fillna(0.0)

        # 2. Technical coefficients A and Leontief inverse L = (I − A)^{-1}
        x_safe = X.replace(0, np.nan)
        A = Z.div(x_safe, axis=1).fillna(0).values
        L = np.linalg.inv(np.eye(len(subs)) - A)

        # 3. LS^cons = ℓ' · L  (row vector: total labour embodied per unit of output)
        return pd.Series(ell.values @ L, index=subs)

    else:
        parent_ls['direct_ls'] = np.where(
            parent_ls['GVA'] > 0,
            parent_ls['e_comp'] / parent_ls['GVA'],
            0.0
        )
        return pd.Series(
            [parent_ls['direct_ls'].get(parent_map.get(s, s), 0.0) for s in subs],
            index=subs
        )


# ==================== ==================== ==================== ====================
# 3. Sectoral LS with continuous weights  (eq. 2.1 in GG-B)

def sectoral_labor_shares(year_result, ls_cons, kappa=0.6):
    """
    Compute LS_C and LS_I using direct final-demand weights and
    consolidated industry labour shares (direct or Leontief-propagated).

        LS_S = sum_i  w_{S,i} * ls_cons_i      for S in {C, I}

    where w_{S,i} = direct final demand from i for S / total direct demand for S.
    The Leontief propagation (if any) lives in ls_cons via consolidated_labor_shares().

    Parameters
    ----------
    year_result : dict   — from compute_direct_for_year()
    ls_cons     : Series — consolidated LS by sub-industry (from above)
    kappa       : float  — capitalisation rate for org services

    Returns
    -------
    dict with keys:
        'LS_C', 'LS_I'  — aggregate sectoral labor shares (ratio)
        'weights_C', 'weights_I' — direct expenditure weights (Series)
        'ls_consolidated' — the input consolidated LS (for inspection)
    """
    out_req = year_result['output_requirements']   # DataFrame, index=sub-industries
    Z       = year_result['Z']

    # 1. output required for C+G and I
    out_C = out_req['C'] + out_req['G']            # consumption = household + govt
    out_I = out_req['I'].copy()

    # 2. organisational services adjustment (60 % rule)
    org_codes = ['69700', '71000', '73000', '74750', '78000', '80820']
    org_in_data = [c for c in org_codes if c in Z.index]

    if org_in_data:
        org_intermediate = Z.loc[org_in_data, :].sum(axis=0)
        org_addition = kappa * org_intermediate.sum()

        # distribute addition proportionally among org sub-industries
        org_mask   = out_I.index.isin(org_in_data)
        org_total  = out_I[org_mask].sum()
        if org_total > 0:
            out_I[org_mask] += org_addition * out_I[org_mask] / org_total

    # 3. continuous weights (normalised to sum to 1)
    w_C = out_C / out_C.sum()
    w_I = out_I / out_I.sum()

    # 4. aggregate sectoral LS
    # align indices
    common = ls_cons.index.intersection(w_C.index)
    LS_C = (w_C[common] * ls_cons[common]).sum()
    LS_I = (w_I[common] * ls_cons[common]).sum()

    return {
        'LS_C': LS_C,
        'LS_I': LS_I,
        'weights_C': w_C,
        'weights_I': w_I,
        'ls_consolidated': ls_cons,
    }


# ==================== ==================== ==================== ====================
# 4. Full time-series

def compute_sectoral_ls_timeseries(years, kappa=0.6,
                                    use_cache=False, cache_dir=CACHE_DIR,
                                    use_leontief=False):
    """
    For each year:
      1. Run direct NX (optionally from cache)
      2. Fetch direct LS from NABP36 (optionally from cache)
      3. Compute consolidated LS
      4. Aggregate with continuous weights

    Returns DataFrame indexed by year with columns:
        LS_C, LS_I, LS_I_minus_C

    Parameters
    ----------
    use_cache : bool
        If True, reads/writes per-year IO pickles and the NABP36 parquet
        instead of hitting the API every run.
    cache_dir : str
        Directory for cache files (shared with direct_NX caches).
    """

    # 1. fetch all labor share data once (optionally cached)
    if use_cache:
        df_ls = load_or_fetch_industry_labor_shares(cache_dir=cache_dir)
    else:
        print("Fetching industry labor shares from NABP36 ...")
        df_ls = fetch_industry_labor_shares()

    rows = []
    for year in years:
        try:
            yr = load_or_compute_year(year, cache_dir=cache_dir) if use_cache \
                 else compute_direct_for_year(year)

            # labor shares for this year (at parent level)
            df_ls_yr = df_ls[df_ls['year'] == year]
            if df_ls_yr.empty:
                print(f"  No NABP36 data for {year}, skipping")
                continue

            # consolidated LS (direct or Leontief-propagated)
            ls_cons = consolidated_labor_shares(yr, df_ls_yr,
                                                use_leontief=use_leontief)

            # aggregate with direct expenditure weights
            res = sectoral_labor_shares(yr, ls_cons, kappa=kappa)

            rows.append({
                'year': year,
                'LS_C': res['LS_C'] * 100,
                'LS_I': res['LS_I'] * 100,
                'LS_I_minus_C': (res['LS_I'] - res['LS_C']) * 100,
            })
            print(f"  {year}:  LS_C={res['LS_C']:.3f}  LS_I={res['LS_I']:.3f}  "
                  f"Δ={res['LS_I']-res['LS_C']:.3f}")

        except Exception as e:
            print(f"  Error for {year}: {e}")
            continue

    df = pd.DataFrame(rows).set_index('year')
    return df


# ==================== ==================== ==================== ====================
# 5. Diagnostic: inspect weights and direct LS for one year

def inspect_year(year, kappa=0.6, use_cache=False, cache_dir=CACHE_DIR):
    """
    Returns a DataFrame aggregated to parent industry level with:
        direct_ls      — CE_j / GVA_j (direct labour share)
        leontief_ls    — ℓ' · L entry (Leontief-consolidated labour share)
        w_C, w_I       — direct expenditure weights (same for both methods)
    Sorted by w_I descending.
    """
    if use_cache:
        df_ls = load_or_fetch_industry_labor_shares(cache_dir=cache_dir)
        yr    = load_or_compute_year(year, cache_dir=cache_dir)
    else:
        df_ls = fetch_industry_labor_shares()
        yr    = compute_direct_for_year(year)
    df_ls_yr = df_ls[df_ls['year'] == year]

    ls_direct = consolidated_labor_shares(yr, df_ls_yr, use_leontief=False)
    ls_leon   = consolidated_labor_shares(yr, df_ls_yr, use_leontief=True)
    res       = sectoral_labor_shares(yr, ls_direct, kappa=kappa)   # weights only

    subs       = yr['X'].index
    parent_map = var_groups.sub_to_parent

    inspect = pd.DataFrame({
        'parent':      [parent_map.get(s, s) for s in subs],
        'direct_ls':   ls_direct.values,
        'leontief_ls': ls_leon.values,
        'w_C':         res['weights_C'].values,
        'w_I':         res['weights_I'].values,
    }, index=subs)

    parent_agg = inspect.groupby('parent').agg({
        'direct_ls':   'mean',
        'leontief_ls': 'mean',
        'w_C':         'sum',
        'w_I':         'sum',
    })

    parent_agg = parent_agg.sort_values('w_I', ascending=False)
    return parent_agg


# ==================== ==================== ==================== ====================
# 6. Plot: LS difference (I − C) over time

def plot_ls_difference(df_ts, save_path='0_output/LS_consolidated.png'):
    """
    Plot LS_I − LS_C over time from the timeseries DataFrame.

    Parameters
    ----------
    df_ts : DataFrame
        Output of compute_sectoral_ls_timeseries(), indexed by year,
        with column 'LS_I_minus_C' (in pp).
    save_path : str or None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))

    ax.plot(
        df_ts.index, df_ts['LS_I_minus_C'],
        color='crimson', lw=2, ls='-',
        label=r'$\Delta LS = LS_I - LS_C$'
    )

    ax.axhline(0, color='#1F2A44', linewidth=1.2, ls='--')
    ax.set_xlim(df_ts.index.min(), df_ts.index.max())
    ax.set_ylabel('Percentage points')
    ax.grid(linewidth=0.6, alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.legend(loc='upper right')

    # fig.suptitle(
    #     'Labour share difference: investment vs consumption sector',
    #     y=0.95
    # )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return fig, ax


# ==================== ==================== ==================== ====================
# 7. Caching helpers

def load_or_fetch_industry_labor_shares(cache_dir=CACHE_DIR, force=False):
    """
    Return fetch_industry_labor_shares(), reading from cache if available.
    Saved to <cache_dir>/nabp36_labor_shares.parquet on first run.
    """
    path = os.path.join(cache_dir, "nabp36_labor_shares.parquet")
    if not force and os.path.exists(path):
        print("  Loading NABP36 labor shares from cache …")
        return pd.read_parquet(path)

    print("Fetching industry labor shares from NABP36 …")
    df = fetch_industry_labor_shares()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved NABP36 to {path}")
    return df


def load_or_compute_ls_timeseries(years, kappa=0.6,
                                   cache_path=None, cache_dir=CACHE_DIR, force=False,
                                   use_leontief=False):
    """
    Return compute_sectoral_ls_timeseries(...), reading from cache if available.

    The timeseries DataFrame is stored as a parquet at cache_path.
    Per-year IO results and NABP36 data are also cached automatically.

    Parameters
    ----------
    years      : iterable of ints
    kappa      : passed through to compute_sectoral_ls_timeseries
    cache_path : explicit parquet path; defaults to
                 <cache_dir>/ls_timeseries_<start>_<end>.parquet
    cache_dir  : directory for all cache files
    force      : if True, ignore existing cache and recompute
    """
    years = list(years)
    if cache_path is None:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{min(years)}_{max(years)}"
        suffix = "_leontief" if use_leontief else ""
        cache_path = os.path.join(cache_dir, f"ls_timeseries_{tag}{suffix}.parquet")

    if not force and os.path.exists(cache_path):
        print(f"Loading LS timeseries from {cache_path} …")
        return pd.read_parquet(cache_path)

    df = compute_sectoral_ls_timeseries(years, kappa=kappa,
                                        use_cache=True, cache_dir=cache_dir,
                                        use_leontief=use_leontief)
    df.to_parquet(cache_path)
    print(f"Saved LS timeseries to {cache_path}")
    return df
