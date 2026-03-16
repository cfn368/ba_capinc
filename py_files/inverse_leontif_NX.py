import pandas as pd
import numpy as np
import py_files.var_groups as var_groups
import matplotlib.pyplot as plt

from numpy.linalg import inv
from dstapi import DstApi


# ========== ========== ========== ========== ========== ==========
# 1. Leontief analysis for a specific year (NX-extended, GG-B methodology)
# ========== ========== ========== ========== ========== ==========
def compute_leontief_for_year(year):
    """
    Complete Leontief analysis for a single year.
    
    Matches GG-B methodology: total final use = C + G + I + (X - M).
    Industry use shares defined relative to this total, so that
    C_share + I_share + NX_share = 100% per industry.
    
    Leontief-adjusted output requirements computed for all four
    final demand categories (C, G, I, X).
    
    Args:
        year: Year to analyze (int or str)
        
    Returns:
        dict with keys: 'year', 'Z', 'X', 'Y', 'L_df', 
        'output_requirements', 'use_shares'
    """
    
    print(f"Processing year {year}...")
    
    # ========== ========== ========== ========== ========== ==========
    # 0. Setup
    
    subgroup_codes = list(var_groups.sub_to_parent.keys())
    tilgang2_detailed = ['T' + code for code in subgroup_codes]
    anvendelse_detailed = ['A' + code for code in subgroup_codes]
    
    industries = subgroup_codes
    n = len(industries)
    
    # ========== ========== ========== ========== ========== ==========
    # 1. Fetch Intermediate Use Matrix Z (industry x industry)
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
    
    # ========== ========== ========== ========== ========== ==========
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
    
    # ========== ========== ========== ========== ========== ==========
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
    
    # ========== ========== ========== ========== ========== ==========
    # 4. Compute net exports and total final use (GG-B definition)
    #    FU_i = C_i + G_i + I_i + (X_i - M_i)
    
    Y['NX'] = Y['X'] - Y['M']
    Y['total_final_use'] = Y['C'] + Y['G'] + Y['I'] + Y['NX']
    
    # ========== ========== ========== ========== ========== ==========
    # 5. Compute Technical Coefficients Matrix A
    
    X_safe = X.replace(0, np.nan)
    A = Z.div(X_safe, axis=1).fillna(0)
    
    # ========== ========== ========== ========== ========== ==========
    # 6. Compute Leontief Inverse L
    
    I_matrix = np.eye(n)
    I_minus_A = I_matrix - A.values
    
    try:
        L = inv(I_minus_A)
        L_df = pd.DataFrame(L, index=industries, columns=industries)
    except np.linalg.LinAlgError:
        print(f"  Warning: Singular matrix for year {year}, using pseudo-inverse")
        L = np.linalg.pinv(I_minus_A)
        L_df = pd.DataFrame(L, index=industries, columns=industries)
    
    # ========== ========== ========== ========== ========== ==========
    # 7. Compute Leontief-adjusted output requirements
    #    for each final demand category
    
    output_for_C = L_df.values @ Y['C'].values
    output_for_G = L_df.values @ Y['G'].values
    output_for_I = L_df.values @ Y['I'].values
    output_for_X = L_df.values @ Y['X'].values
    
    output_requirements = pd.DataFrame({
        'C': output_for_C,
        'G': output_for_G,
        'I': output_for_I,
        'X': output_for_X,
    }, index=industries)
    
    # GG-B: total final use = C + G + I + NX
    # For Leontief-adjusted totals, this is output_for_(C+G+I+X) - imports
    # But since we want *domestic* output shares, the Leontief inverse
    # already captures domestic production only. The total domestic output
    # needed to satisfy all final demand (incl. exports) is:
    output_requirements['total_final_use'] = (
        output_requirements['C'] + 
        output_requirements['G'] + 
        output_requirements['I'] + 
        output_requirements['X']
    )
    
    # Also keep domestic-only for backward compatibility
    output_requirements['total_domestic'] = (
        output_requirements['C'] + 
        output_requirements['G'] + 
        output_requirements['I']
    )
    
    # ========== ========== ========== ========== ========== ==========
    # 8. Compute Use Shares (Leontief-Adjusted, GG-B definition)
    #    Denominator = total final use including exports
    
    use_shares = pd.DataFrame(index=industries)
    
    # --- GG-B shares: denominator includes exports ---
    # Consumption share: (C + G) / (C + G + I + X)
    use_shares['C_share'] = (
        (output_requirements['C'] + output_requirements['G']) / 
        output_requirements['total_final_use'] * 100
    )
    
    # Investment share: I / (C + G + I + X)
    use_shares['I_share'] = (
        output_requirements['I'] / 
        output_requirements['total_final_use'] * 100
    )
    
    # Export share: X / (C + G + I + X)
    use_shares['X_share'] = (
        output_requirements['X'] / 
        output_requirements['total_final_use'] * 100
    )
    
    # --- Direct (non-Leontief) shares for comparison ---
    total_direct = Y['C'] + Y['G'] + Y['I'] + Y['X']
    total_direct_safe = total_direct.replace(0, np.nan)
    
    use_shares['C_direct'] = ((Y['C'] + Y['G']) / total_direct_safe * 100)
    use_shares['I_direct'] = (Y['I'] / total_direct_safe * 100)
    use_shares['X_direct'] = (Y['X'] / total_direct_safe * 100)
    
    # --- NX-based shares matching GG-B Table A1 ---
    # These use NX = X - M in denominator, giving shares that sum to 100%
    # FU = C + G + I + NX
    fu = Y['total_final_use']
    fu_safe = fu.replace(0, np.nan)
    
    use_shares['C_share_NX'] = ((Y['C'] + Y['G']) / fu_safe * 100)
    use_shares['I_share_NX'] = (Y['I'] / fu_safe * 100)
    use_shares['NX_share'] = (Y['NX'] / fu_safe * 100)
    
    # Add output for weighting
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
        'X': X,
        'Y': Y,
        'L_df': L_df,
        'output_requirements': output_requirements,
        'use_shares': use_shares,
    }


# ========== ========== ========== ========== ========== ==========   
#  2. Classify by investment type with organizational services adjustment
# ========== ========== ========== ========== ========== ==========
def classify_investment_by_type(year_result, kappa=0.6):
    """
    Classify investment by type for a single year result.
    Apply organizational services adjustment.
    
    Args:
        year_result: Output from compute_leontief_for_year()
        kappa: Capitalization rate for organizational services (default 0.6)
        
    Returns:
        dict with investment shares by type
    """
    
    year = year_result['year']
    use_shares = year_result['use_shares'].copy()
    Z = year_result['Z']
    Y = year_result['Y']
    
    # Add classifications
    use_shares['tangible_intangible'] = use_shares.index.map(var_groups.industry_classification)
    use_shares['investment_type'] = use_shares.index.map(var_groups.investment_type)
    
    # ========== ========== ========== ========== ========== ==========
    # 1. Handle organizational services (kappa rule)
    
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
    
    # ========== ========== ========== ========== ========== ==========
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


# ========== ========== ========== ========== ========== ==========
# 3. GDP data
# ========== ========== ========== ========== ========== ==========
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


# ========== ========== ========== ========== ========== ==========
# 4. Timeseries
# ========== ========== ========== ========== ========== ==========
def compute_investment_timeseries(years, kappa=0.6, normalize_by_gdp=True):
    """
    Compute investment composition over multiple years.
    """
    
    results = []
    
    for year in years:
        try:
            year_result = compute_leontief_for_year(year)
            investment_shares = classify_investment_by_type(year_result, kappa)
            results.append(investment_shares)
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue
    
    df = pd.DataFrame(results)
    df = df.set_index('year')
    
    if normalize_by_gdp:
        print("\nFetching GDP data for normalization...")
        gdp = fetch_gdp_data(df.index)
        
        investment_cols = ['structures', 'equipment', 'intellectual_property', 'organizational']
        
        for col in investment_cols:
            df[col] = df['total_investment'] * df[col] / 100
        
        df['tangible'] = df['structures'] + df['equipment']
        df['intangible'] = df['intellectual_property'] + df['organizational']
        
        for col in investment_cols + ['tangible', 'intangible']:
            df[col] = (df[col] / gdp) * 100
        
        print("  Normalized by GDP")
    
    return df


# ========== ========== ========== ========== ========== ==========
# 5. Plot as fraction of GDP
# ========== ========== ========== ========== ========== ==========
def plot_investment_composition(investment_timeseries, as_pct_gdp=True):
    """
    Create stacked area plot of investment composition.
    """

    wide = investment_timeseries.copy()
    
    order = ["structures", "equipment", "intellectual_property", "organizational"]
    labels = ["Structure", "Equipment", "Intellectual Property Products", "Organizational services"]
    
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
    
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    polys1 = ax1.stackplot(
        x, ys_left, colors=colors_left, linewidth=0.8, alpha=0.85
    )
    
    polys2 = ax2.stackplot(
        x, ys_right, colors=colors_right, linewidth=0.8, alpha=0.85
    )
    
    ymax = float(np.nanmax(ys_left.sum(axis=0)))
    
    if as_pct_gdp:
        ylim_top = 50
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
    plt.savefig('0_output/I_decomp.png', dpi=200)
    plt.show()
    

# ========== ========== ========== ========== ========== ==========
# 6. Aggregate use shares to parent level
# ========== ========== ========== ========== ========== ==========
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