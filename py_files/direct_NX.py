import os
import pickle
import pandas as pd
import numpy as np
import py_files.var_groups as var_groups
import matplotlib.pyplot as plt

from dstapi import DstApi

CACHE_DIR = "0_intermediate"


# ==================== ==================== ==================== ====================
# 1. Direct final-demand analysis for a specific year (NX-extended, GG-B methodology)

def compute_direct_for_year(year):
    """
    Direct final-demand analysis for a single year.

    Args:
        year: Year to analyze (int or str)

    Returns:
        dict with keys: 'year', 'Z', 'X', 'Y',
                        'output_requirements', 'use_shares'
        (no 'L_df' key — no Leontief inverse is computed)
    """

    print(f"Processing year {year}...")

    # 0. Setup

    subgroup_codes = list(var_groups.sub_to_parent.keys())
    tilgang2_detailed = ['T' + code for code in subgroup_codes]
    anvendelse_detailed = ['A' + code for code in subgroup_codes]

    industries = subgroup_codes
    n = len(industries)

    # 1. Fetch Intermediate Use Matrix Z (industry x industry)
    #    Still needed for the organisational-services kappa adjustment.

    NAIO1F = DstApi('NAIO1F')

    params_Z = {
        'table': 'NAIO1F',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED',   'values': ['V']},
            {'code': 'Tid',         'values': [str(year)]},
            {'code': 'TILGANG1',    'values': ['P1_BP']},
            {'code': 'TILGANG2',    'values': tilgang2_detailed},
            {'code': 'ANVENDELSE',  'values': anvendelse_detailed},
        ]
    }

    io_data = NAIO1F.get_data(params=params_Z)
    io_data['INDHOLD'] = pd.to_numeric(io_data['INDHOLD'], errors='coerce')

    io_data['supply_code'] = io_data['TILGANG2'].str.split(' ').str[0]
    io_data['use_code'] = io_data['ANVENDELSE'].str.split(' ').str[0]

    Z = pd.DataFrame(0.0, index=industries, columns=industries)

    io_intermediate = io_data[
        io_data['supply_code'].isin(industries) &
        io_data['use_code'].isin(industries)
    ]

    for _, row in io_intermediate.iterrows():
        i = row['supply_code']
        j = row['use_code']
        Z.loc[i, j] = row['INDHOLD']

    # 1b. Fetch imported intermediates and build total-use Z
    #     Z_total[i,j] = domestic + imported use of product type i by industry j.
    #     With total-use Z: col_sum(A_total) = (X - GVA) / X, so v'·L = 1
    #     by accounting identity — no import-leakage correction needed downstream.

    params_Z_M = {
        'table': 'NAIO1F',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED',   'values': ['V']},
            {'code': 'Tid',         'values': [str(year)]},
            {'code': 'TILGANG1',    'values': ['P7AD2121']},
            {'code': 'TILGANG2',    'values': tilgang2_detailed},
            {'code': 'ANVENDELSE',  'values': anvendelse_detailed},
        ]
    }

    io_data_M = NAIO1F.get_data(params=params_Z_M)
    io_data_M['INDHOLD'] = pd.to_numeric(io_data_M['INDHOLD'], errors='coerce')
    io_data_M['supply_code'] = io_data_M['TILGANG2'].str.split(' ').str[0]
    io_data_M['use_code']    = io_data_M['ANVENDELSE'].str.split(' ').str[0]

    Z_M = pd.DataFrame(0.0, index=industries, columns=industries)

    io_intermediate_M = io_data_M[
        io_data_M['supply_code'].isin(industries) &
        io_data_M['use_code'].isin(industries)
    ]

    for _, row in io_intermediate_M.iterrows():
        i = row['supply_code']
        j = row['use_code']
        Z_M.loc[i, j] = row['INDHOLD']

    Z_total = Z + Z_M

    # 2. Fetch Total Output X

    params_X = {
        'table': 'NAIO1F',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED',   'values': ['V']},
            {'code': 'Tid',         'values': [str(year)]},
            {'code': 'TILGANG1',    'values': ['P1_BP']},
            {'code': 'TILGANG2',    'values': tilgang2_detailed},
            {'code': 'ANVENDELSE',  'values': ['AA00000']},
        ]
    }

    output_data = NAIO1F.get_data(params=params_X)
    output_data['INDHOLD'] = pd.to_numeric(output_data['INDHOLD'], errors='coerce')
    output_data['code'] = output_data['TILGANG2'].str.split(' ').str[0]

    X = output_data.set_index('code')['INDHOLD'].reindex(industries).fillna(0)

    # 3. Fetch Final Demand Y (C, G, I, X, M)

    final_demand_codes = {
        'C': 'ACPT',    # Household consumption
        'G': 'ACO',     # Government consumption
        'I': 'ABI',     # Gross fixed capital formation
        'X': 'AE6000',  # Exports
    }

    Y_dict = {}

    for category, code in final_demand_codes.items():
        params_Y = {
            'table': 'NAIO1F',
            'format': 'BULK',
            'lang': 'en',
            'variables': [
                {'code': 'PRISENHED',   'values': ['V']},
                {'code': 'Tid',         'values': [str(year)]},
                {'code': 'TILGANG1',    'values': ['P1_BP']},
                {'code': 'TILGANG2',    'values': tilgang2_detailed},
                {'code': 'ANVENDELSE',  'values': [code]},
            ]
        }

        y_data = NAIO1F.get_data(params=params_Y)
        y_data['INDHOLD'] = pd.to_numeric(y_data['INDHOLD'], errors='coerce')
        y_data['code'] = y_data['TILGANG2'].str.split(' ').str[0]

        Y_dict[category] = y_data.set_index('code')['INDHOLD'].reindex(industries).fillna(0)

    # Fetch imports
    params_M = {
        'table': 'NAIO1F',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED',   'values': ['V']},
            {'code': 'Tid',         'values': [str(year)]},
            {'code': 'TILGANG1',    'values': ['P7AD2121']},
            {'code': 'TILGANG2',    'values': tilgang2_detailed},
            {'code': 'ANVENDELSE',  'values': ['AIAE']},
        ]
    }

    m_data = NAIO1F.get_data(params=params_M)
    m_data['INDHOLD'] = pd.to_numeric(m_data['INDHOLD'], errors='coerce')
    m_data['code'] = m_data['TILGANG2'].str.split(' ').str[0]

    Y_dict['M'] = m_data.set_index('code')['INDHOLD'].reindex(industries).fillna(0)

    Y = pd.DataFrame(Y_dict, index=industries)

    # 4. Net exports and total final use (GG-B definition)
    #    FU_i = C_i + G_i + I_i + (X_i - M_i)

    Y['NX'] = Y['X'] - Y['M']
    Y['total_final_use'] = Y['C'] + Y['G'] + Y['I'] + Y['NX']

    # 5. Direct output requirements
    #    No Leontief multiplier — each cell is just the IO final-demand entry.
    #    output_requirements[i, j] = amount industry i directly sold to demand j.

    output_requirements = pd.DataFrame({
        'C': Y['C'].values,
        'G': Y['G'].values,
        'I': Y['I'].values,
        'X': Y['X'].values,
    }, index=industries)

    output_requirements['total_final_use'] = (
        output_requirements['C'] +
        output_requirements['G'] +
        output_requirements['I'] +
        output_requirements['X']
    )

    output_requirements['total_domestic'] = (
        output_requirements['C'] +
        output_requirements['G'] +
        output_requirements['I']
    )

    # 6. Use Shares (direct, GG-B definition)

    use_shares = pd.DataFrame(index=industries)

    total_direct = output_requirements['total_final_use']
    total_direct_safe = total_direct.replace(0, np.nan)

    # 1. Main shares (C+G, I, X out of C+G+I+X)
    use_shares['C_share'] = (
        (output_requirements['C'] + output_requirements['G']) /
        total_direct_safe * 100
    )
    use_shares['I_share'] = (
        output_requirements['I'] /
        total_direct_safe * 100
    )
    use_shares['X_share'] = (
        output_requirements['X'] /
        total_direct_safe * 100
    )

    # 2. Direct shares (identical to above in the no-Leontief case, kept for API compat)
    use_shares['C_direct'] = use_shares['C_share']
    use_shares['I_direct'] = use_shares['I_share']
    use_shares['X_direct'] = use_shares['X_share']

    # 3. NX-based shares matching GG-B Table A1
    fu = Y['total_final_use']
    fu_safe = fu.replace(0, np.nan)

    use_shares['C_share_NX'] = ((Y['C'] + Y['G']) / fu_safe * 100)
    use_shares['I_share_NX'] = (Y['I'] / fu_safe * 100)
    use_shares['NX_share'] = (Y['NX'] / fu_safe * 100)

    # 4. Output columns for weighting / downstream use
    use_shares['output'] = X
    use_shares['output_for_investment'] = output_requirements['I']
    use_shares['output_for_consumption'] = (
        output_requirements['C'] + output_requirements['G']
    )
    use_shares['output_for_exports'] = output_requirements['X']
    use_shares['output_total_final_use'] = output_requirements['total_final_use']

    return {
        'year': year,
        'Z': Z,
        'Z_total': Z_total,
        'X': X,
        'Y': Y,
        'output_requirements': output_requirements,
        'use_shares': use_shares,
    }
    

# ==================== ==================== ==================== ====================
# 3. Classify by investment type with organisational services adjustment

def classify_investment_by_type(year_result, kappa=0.6):
    """
    Classify investment by type for a single year result.
    Apply organisational services adjustment.

    The kappa rule adds kappa * (total intermediate purchases from org-service
    industries) to investment — this uses Z directly and is unrelated to the
    Leontief inverse, so it is retained here.

    Args:
        year_result: Output from compute_direct_for_year()
        kappa: Capitalisation rate for organisational services (default 0.6)

    Returns:
        dict with investment shares by type
    """

    year = year_result['year']
    use_shares = year_result['use_shares'].copy()
    Z = year_result['Z']

    # Add classifications
    use_shares['tangible_intangible'] = use_shares.index.map(var_groups.industry_classification)
    use_shares['investment_type'] = use_shares.index.map(var_groups.investment_type)

    # 1. Handle organisational services (kappa rule)

    org_codes = ['69700', '71000', '73000', '74750', '78000', '80820']
    org_codes_in_data = [c for c in org_codes if c in Z.index]

    if org_codes_in_data:
        org_intermediate = Z.loc[org_codes_in_data, :].sum(axis=0)
        total_org_intermediate = org_intermediate.sum()
        org_investment_addition = kappa * total_org_intermediate

        org_mask = use_shares['investment_type'] == 'organizational'
        org_output_total = use_shares.loc[org_mask, 'output_for_investment'].sum()

        if org_output_total > 0:
            use_shares.loc[org_mask, 'output_for_investment'] += (
                org_investment_addition *
                use_shares.loc[org_mask, 'output_for_investment'] /
                org_output_total
            )

    # 2. Aggregate by investment type

    investment_by_type = use_shares.groupby('investment_type')['output_for_investment'].sum()
    total_investment = investment_by_type.sum()

    shares = {
        'year': year,
        'structures': (investment_by_type.get('structures', 0) / total_investment * 100) if total_investment > 0 else 0,
        'equipment': (investment_by_type.get('equipment', 0) / total_investment * 100) if total_investment > 0 else 0,
        'intellectual_property': (investment_by_type.get('intellectual_property', 0) / total_investment * 100) if total_investment > 0 else 0,
        'organizational': (investment_by_type.get('organizational', 0) / total_investment * 100) if total_investment > 0 else 0,
    }

    shares['tangible'] = shares['structures'] + shares['equipment']
    shares['intangible'] = shares['intellectual_property'] + shares['organizational']
    shares['total_investment'] = total_investment

    return shares


# ==================== ==================== ==================== ====================
# 3b. GDP data

def fetch_gdp_data(years):
    """
    Fetch GDP data from Danish Statistics for specified years.
    """

    NAN1 = DstApi('NAN1')

    params_Y = {
        'table': 'NAN1',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'PRISENHED', 'values': ['V_M']},
            {'code': 'Tid', 'values': [str(y) for y in years]},
            {'code': 'TRANSAKT', 'values': ['B1GQK']}
        ]
    }

    df_Y = NAN1.get_data(params=params_Y)
    df_Y['INDHOLD'] = pd.to_numeric(df_Y['INDHOLD'], errors='coerce')

    GDP = (df_Y.loc[df_Y['TRANSAKT'].eq('B.1*g Gross domestic product'), ['TID', 'INDHOLD']]
             .groupby('TID', as_index=False)['INDHOLD'].sum()
             .rename(columns={'INDHOLD': 'GDP', 'TID': 'year'}))

    GDP['GDP'] = GDP['GDP'] * 1e3
    GDP['year'] = GDP['year'].astype(int)

    return GDP.set_index('year')['GDP']


# ==================== ==================== ==================== ====================
# 4. Timeseries

def compute_investment_timeseries(years, kappa=0.6, normalize_by_gdp=True,
                                   use_cache=False, cache_dir=CACHE_DIR):
    """
    Compute investment composition over multiple years.
    Set use_cache=True to read/write per-year pickles instead of hitting the API.
    """

    results = []

    for year in years:
        try:
            if use_cache:
                year_result = load_or_compute_year(year, cache_dir=cache_dir)
            else:
                year_result = compute_direct_for_year(year)
            investment_shares = classify_investment_by_type(year_result, kappa)
            results.append(investment_shares)
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue

    df = pd.DataFrame(results)
    df = df.set_index('year')

    if normalize_by_gdp:
        print("\nFetching GDP data for normalisation...")
        gdp = fetch_gdp_data(df.index)

        investment_cols = ['structures', 'equipment', 'intellectual_property', 'organizational']

        for col in investment_cols:
            df[col] = df['total_investment'] * df[col] / 100

        df['tangible'] = df['structures'] + df['equipment']
        df['intangible'] = df['intellectual_property'] + df['organizational']

        for col in investment_cols + ['tangible', 'intangible']:
            df[col] = (df[col] / gdp) * 100

        print("  Normalised by GDP")

    return df


# ==================== ==================== ==================== ====================
# 5. Plot as fraction of GDP

def plot_investment_composition(investment_timeseries, as_pct_gdp=True):
    """
    Create stacked area plot of investment composition.
    """

    wide = investment_timeseries.copy()

    order = ["structures", "equipment", "intellectual_property", "organizational"]
    labels = ["Structure", "Equipment", "Intellectual Property Products", "Organisational services"]

    ys_left = np.vstack([wide[c].to_numpy() for c in order])
    x = wide.index.to_numpy()

    colors_left = [
        "#9E1B32",
        "#1F2A44",
        "#D6C3A5",
        "#2A9D8F",
    ]

    tangible = wide["tangible"].to_numpy()
    intangible = wide["intangible"].to_numpy()

    ys_right = np.vstack([tangible, intangible])
    labels_right = ["Tangible", "Intangible"]
    colors_right = [
        "#1F2A44",
        "#2A9D8F",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    polys1 = ax1.stackplot(
        x, ys_left, colors=colors_left, linewidth=0.8, alpha=0.85
    )

    polys2 = ax2.stackplot(
        x, ys_right, colors=colors_right, linewidth=0.8, alpha=0.85
    )

    ymax = float(np.nanmax(ys_left.sum(axis=0)))

    if as_pct_gdp:
        ylim_top = 30
        tick_interval = 10
    else:
        ylim_top = min(100, ymax * 1.3)
        tick_interval = 10

    for ax in (ax1, ax2):
        ax.set_ylim(0, ylim_top)
        ticks = np.arange(0, int(ylim_top) + 1, tick_interval)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.0f}%" for t in ticks])
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)

    ax1.set_title("Four aggregate capital types", pad=15)
    ax2.set_title("Investment: tangible vs intangible", pad=15)

    ylabel = "Percent of GDP" if as_pct_gdp else "Percent of total investment"
    ax2.set_ylabel(ylabel)

    ax1.legend(handles=polys1[::-1], labels=labels[::-1],
               frameon=True, loc="upper left")
    ax2.legend(handles=polys2[::-1], labels=labels_right[::-1],
               frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig('0_output/I_decomp_direct.png', dpi=200)
    plt.show()


# ==================== ==================== ==================== ====================
# 6. Aggregate use shares to parent level

def aggregate_use_shares_to_parent(year_result):
    """
    Aggregate use shares to parent industry level.
    """

    use_shares = year_result['use_shares'].copy()
    use_shares['parent'] = use_shares.index.map(var_groups.sub_to_parent)

    parent_shares = use_shares.groupby('parent').apply(
        lambda x: pd.Series({
            'C_share': np.average(x['C_share'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'I_share': np.average(x['I_share'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'X_share': np.average(x['X_share'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'C_direct': np.average(x['C_direct'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'I_direct': np.average(x['I_direct'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'X_direct': np.average(x['X_direct'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'C_share_NX': np.average(x['C_share_NX'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'I_share_NX': np.average(x['I_share_NX'], weights=x['output']) if x['output'].sum() > 0 else 0,
            'NX_share': np.average(x['NX_share'], weights=x['output']) if x['output'].sum() > 0 else 0,
        })
    ).reset_index()

    return parent_shares


# ==================== ==================== ==================== ====================
# 7. Caching helpers

def _year_pickle_path(year, cache_dir):
    return os.path.join(cache_dir, f"year_{year}.pkl")


def load_or_compute_year(year, cache_dir=CACHE_DIR, force=False):
    """
    Return compute_direct_for_year(year), reading from cache if available.

    Parameters
    ----------
    year       : int
    cache_dir  : directory for pickle files (created if missing)
    force      : if True, re-fetch from API even when a cache file exists

    The result is saved to  <cache_dir>/year_<year>.pkl  on first run.
    """
    path = _year_pickle_path(year, cache_dir)
    if not force and os.path.exists(path):
        print(f"  Loading year {year} from cache …")
        with open(path, "rb") as f:
            return pickle.load(f)

    result = compute_direct_for_year(year)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved year {year} to cache.")
    return result


def load_or_compute_timeseries(years, kappa=0.6, normalize_by_gdp=True,
                                cache_path=None, cache_dir=CACHE_DIR, force=False):
    """
    Return compute_investment_timeseries(...), reading from cache if available.

    The timeseries DataFrame is stored as a single parquet file at cache_path.
    Per-year raw results are also cached as pickles (via load_or_compute_year),
    so individual years are never re-fetched.

    Parameters
    ----------
    years           : iterable of ints
    kappa           : capitalisation rate passed to classify_investment_by_type
    normalize_by_gdp: passed through
    cache_path      : explicit parquet path; defaults to
                      <cache_dir>/timeseries_<start>_<end>.parquet
    cache_dir       : directory for per-year pickles
    force           : if True, ignore existing cache files and recompute
    """
    years = list(years)
    if cache_path is None:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{min(years)}_{max(years)}"
        cache_path = os.path.join(cache_dir, f"timeseries_{tag}.parquet")

    if not force and os.path.exists(cache_path):
        print(f"Loading timeseries from {cache_path} …")
        return pd.read_parquet(cache_path)

    df = compute_investment_timeseries(
        years, kappa=kappa, normalize_by_gdp=normalize_by_gdp,
        use_cache=True, cache_dir=cache_dir,
    )
    df.to_parquet(cache_path)
    print(f"Saved timeseries to {cache_path}")
    return df
