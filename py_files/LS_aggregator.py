"""
Sectoral labor share: GG-B methodology applied to Danish data
=============================================================

Two corrections relative to prior approach:
  1. Consolidated labor shares (Leontief-adjusted) instead of direct LS
  2. Continuous expenditure weights instead of binary industry classification

References: Gomez & Gouin-Bonenfant (2025), Section 2.2, eq. (2.1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import py_files.var_groups as var_groups

from py_files.inverse_leontif import compute_leontief_for_year
from dstapi import DstApi


# ========== ========== ========== ========== ========== ==========
# 1. Fetch direct labor shares from NABP36 (industry × year)
# ========== ========== ========== ========== ========== ==========
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


# ========== ========== ========== ========== ========== ==========
# 2. Consolidated labor share per industry (Leontief-adjusted)
# ========== ========== ========== ========== ========== ==========
def consolidated_labor_shares(year_result, df_ls_year):
    """
    Compute the consolidated labor share for each sub-industry
    using the Leontief inverse from `year_result`.

    Consolidated LS of industry i:
        LS^cons_i = sum_j  L_{ji} * (D1_j / X_j)

    where L is the Leontief inverse, D1_j is compensation in j,
    and X_j is gross output of j.

    Parameters
    ----------
    year_result : dict
        Output of compute_leontief_for_year().  Uses keys:
        'L_df'  — Leontief inverse  (sub-industries × sub-industries)
        'X'     — gross output vector (sub-industries)
    df_ls_year : DataFrame
        Labour-share data for a *single year*, with columns
        'branche_code' (36a2 parent code), 'e_comp', 'GVA'.

    Returns
    -------
    Series indexed by sub-industry code with consolidated LS (ratio, not %).
    """
    L_df = year_result['L_df']                # (n_sub × n_sub)
    X    = year_result['X']                   # (n_sub,)
    subs = L_df.index                         # sub-industry codes

    # --- map each sub to its 36a2 parent ---
    parent_map = var_groups.sub_to_parent      # {sub_code: parent_code}

    # --- build direct labor-to-output ratio for every sub-industry ---
    #     We only observe D1 and GVA at the parent level (NABP36),
    #     so every sub inherits its parent's direct LS.
    parent_ls = (
        df_ls_year
        .set_index('branche_code')
    )
    # direct LS at parent level (ratio)
    parent_ls['direct_ls'] = np.where(
        parent_ls['GVA'] > 0,
        parent_ls['e_comp'] / parent_ls['GVA'],
        0.0
    )

    # map to sub-industries
    direct_ls_sub = pd.Series(
        [parent_ls['direct_ls'].get(parent_map.get(s, s), 0.0) for s in subs],
        index=subs
    )

    # --- but we need labor / output, not labor / GVA ---
    #     direct_ls gives D1/GVA.  We need D1/X for the Leontief weighting.
    #     Relationship:  D1_i / X_i  =  (D1_i / GVA_i) * (GVA_i / X_i)
    #     We approximate GVA_i / X_i from the IO table:
    #         GVA_i = X_i - sum_j Z_{ji}   (output minus intermediate inputs)
    Z = year_result['Z']
    intermediate_inputs = Z.sum(axis=0)          # column sums = total intermediate use by industry
    va_ratio = np.where(
        X > 0,
        (X - intermediate_inputs) / X,
        0.0
    )
    va_ratio = pd.Series(va_ratio, index=subs)

    # direct labor-to-output ratio
    labor_output_ratio = direct_ls_sub * va_ratio    # D1_i / X_i

    # --- consolidated LS: sum_j L_{ji} * (D1_j / X_j)  for each i ---
    #     In matrix form:  LS^cons = L^T @ labor_output_ratio
    ls_consolidated = L_df.values.T @ labor_output_ratio.values
    ls_consolidated = pd.Series(ls_consolidated, index=subs)

    return ls_consolidated


# ========== ========== ========== ========== ========== ==========
# 3. Sectoral LS with continuous weights  (eq. 2.1 in GG-B)
# ========== ========== ========== ========== ========== ==========
def sectoral_labor_shares(year_result, ls_cons, kappa=0.6):
    """
    Compute LS_C and LS_I using continuous Leontief-derived weights
    and consolidated industry labor shares.

        LS_S = sum_i  w_{S,i} * LS^cons_i      for S in {C, I}

    where w_{S,i} = (output required from i for S) / (total output for S).

    The organizational-services adjustment (kappa rule) is applied
    to the I-sector output requirements before computing weights,
    exactly as in the investment composition code.

    Parameters
    ----------
    year_result : dict   — from compute_leontief_for_year()
    ls_cons     : Series — consolidated LS by sub-industry (from above)
    kappa       : float  — capitalisation rate for org services

    Returns
    -------
    dict with keys:
        'LS_C', 'LS_I'  — aggregate sectoral labor shares (ratio)
        'weights_C', 'weights_I' — continuous industry weights (Series)
        'ls_consolidated' — the input consolidated LS (for inspection)
    """
    out_req = year_result['output_requirements']   # DataFrame, index=sub-industries
    Z       = year_result['Z']

    # --- output required for C+G and I ---
    out_C = out_req['C'] + out_req['G']            # consumption = household + govt
    out_I = out_req['I'].copy()

    # --- organisational services adjustment (60 % rule) ---
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

    # --- continuous weights (normalised to sum to 1) ---
    w_C = out_C / out_C.sum()
    w_I = out_I / out_I.sum()

    # --- aggregate sectoral LS ---
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


# ========== ========== ========== ========== ========== ==========
# 4. Full time-series
# ========== ========== ========== ========== ========== ==========
def compute_sectoral_ls_timeseries(years, kappa=0.6):
    """
    For each year:
      1. Run Leontief (unchanged)
      2. Fetch direct LS from NABP36
      3. Compute consolidated LS
      4. Aggregate with continuous weights

    Returns DataFrame indexed by year with columns:
        LS_C, LS_I, LS_I_minus_C
    """

    # --- fetch all labor share data once ---
    print("Fetching industry labor shares from NABP36 ...")
    df_ls = fetch_industry_labor_shares()

    rows = []
    for year in years:
        try:
            # Leontief (at sub-industry level)
            yr = compute_leontief_for_year(year)

            # labor shares for this year (at parent level)
            df_ls_yr = df_ls[df_ls['year'] == year]
            if df_ls_yr.empty:
                print(f"  No NABP36 data for {year}, skipping")
                continue

            # consolidated LS
            ls_cons = consolidated_labor_shares(yr, df_ls_yr)

            # aggregate with continuous weights
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


# ========== ========== ========== ========== ========== ==========
# 5. Diagnostic: inspect weights and consolidated LS for one year
# ========== ========== ========== ========== ========== ==========
def inspect_year(year, kappa=0.6):
    """
    Returns a DataFrame at the sub-industry level with:
        parent, direct_ls, consolidated_ls, w_C, w_I
    Useful for sanity-checking individual industries (e.g. pharma).
    """
    df_ls = fetch_industry_labor_shares()
    yr = compute_leontief_for_year(year)
    df_ls_yr = df_ls[df_ls['year'] == year]

    ls_cons = consolidated_labor_shares(yr, df_ls_yr)
    res     = sectoral_labor_shares(yr, ls_cons, kappa=kappa)

    subs = yr['L_df'].index
    parent_map = var_groups.sub_to_parent

    parent_ls = df_ls_yr.set_index('branche_code')
    parent_ls['direct_ls'] = np.where(
        parent_ls['GVA'] > 0,
        parent_ls['e_comp'] / parent_ls['GVA'],
        0.0
    )

    inspect = pd.DataFrame({
        'parent':           [parent_map.get(s, s) for s in subs],
        'direct_ls':        [parent_ls['direct_ls'].get(parent_map.get(s, s), np.nan) for s in subs],
        'consolidated_ls':  ls_cons,
        'w_C':              res['weights_C'],
        'w_I':              res['weights_I'],
    }, index=subs)

    # aggregate to parent level for readability
    parent_agg = inspect.groupby('parent').agg({
        'direct_ls':        'first',
        'consolidated_ls':  lambda x: np.average(x, weights=res['weights_C'].reindex(x.index).fillna(0) +
                                                           res['weights_I'].reindex(x.index).fillna(0))
                                      if (res['weights_C'].reindex(x.index).fillna(0) +
                                          res['weights_I'].reindex(x.index).fillna(0)).sum() > 0
                                      else x.mean(),
        'w_C':              'sum',
        'w_I':              'sum',
    })

    parent_agg = parent_agg.sort_values('w_I', ascending=False)
    return parent_agg


# ========== ========== ========== ========== ========== ==========
# 6. Plot: LS difference (I − C) over time
# ========== ========== ========== ========== ========== ==========
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
        label=r'$\Delta LS$ (investment - consumption sector)'
    )

    ax.axhline(0, color='#1F2A44', linewidth=1.2, ls='--')
    ax.set_xlim(df_ts.index.min(), df_ts.index.max())
    ax.set_ylabel('%')
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


# ========== ========== ========== ========== ========== ==========
# 7. Plot: LS levels for both sectors over time
# ========== ========== ========== ========== ========== ==========
def plot_ls_levels(df_ts, save_path='0_output/LS_levels.png'):
    """
    Two-panel plot showing LS_C and LS_I over time (like GG-B Fig 4).

    Parameters
    ----------
    df_ts : DataFrame from compute_sectoral_ls_timeseries()
    save_path : str or None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # --- consumption sector ---
    ax1.stackplot(
        df_ts.index, df_ts['LS_C'],
        colors=['#1F2A44'], alpha=0.65, linewidth=0.8
    )
    ax1.set_title('Aggregate consumption sector', pad=12)
    ax1.set_ylabel('Labour share (%)')
    ax1.set_ylim(50, 70)

    # --- investment sector ---
    ax2.stackplot(
        df_ts.index, df_ts['LS_I'],
        colors=['#2A9D8F'], alpha=0.65, linewidth=0.8
    )
    ax2.set_title('Aggregate investment sector', pad=12)

    for ax in (ax1, ax2):
        ax.set_xlim(df_ts.index.min(), df_ts.index.max())
        ax.grid(linewidth=0.6, alpha=0.35)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ticks = np.arange(0, 81, 20)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.0f}%" for t in ticks])

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, dpi=200)
    plt.show()

    return fig, (ax1, ax2)


# ========== ========== ========== ========== ========== ==========
# 8. Plot: LS difference with multiple variations (overlay)
# ========== ========== ========== ========== ========== ==========
def plot_ls_variations(variations: dict, colors=None,
                       save_path='0_output/LS_variations.png'):
    """
    Overlay multiple LS_I − LS_C series on one axis, e.g. to compare
    kappa values or org-services inclusion.

    Parameters
    ----------
    variations : dict
        {label: DataFrame} where each DataFrame is the output of
        compute_sectoral_ls_timeseries() (indexed by year).
    colors : list or None
    save_path : str or None

    Example
    -------
    >>> base = sls.compute_sectoral_ls_timeseries(years, kappa=0.6)
    >>> no_org = sls.compute_sectoral_ls_timeseries(years, kappa=0.0)
    >>> sls.plot_ls_variations({
    ...     r'$\kappa = 0.6$': base,
    ...     r'$\kappa = 0$ (no org. adj.)': no_org,
    ... })
    """
    if colors is None:
        colors = ['#F76A4D', '#41FAB4', '#4D9FF7', '#F7D94D', '#A44DF7']

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    for (label, df_ts), color in zip(variations.items(), colors):
        ax.plot(
            df_ts.index, df_ts['LS_I_minus_C'],
            color=color, lw=2, ls='-',
            label=label
        )

    ax.axhline(0, color='0.2', linewidth=2, ls='--')

    # x-limits from widest series
    all_years = np.concatenate([df.index.values for df in variations.values()])
    ax.set_xlim(all_years.min(), all_years.max())

    ax.set_ylabel(r'$\Delta LS \;(I-C)\;$ pp.')
    ax.grid(linewidth=0.6, alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.legend(loc='lower left')

    fig.suptitle(
        'Labour share difference: investment vs consumption sector',
        y=0.95
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return fig, ax