import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_labor_share_variations(df_e_slim, variations: dict, colors=None):
    if colors is None:
        colors = ['#F76A4D', '#41FAB4', '#4D9FF7', '#F7D94D', '#A44DF7']

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    results = {}

    for (label, sector_dicts), color in zip(variations.items(), colors):
        c_industries = sector_dicts['C']
        i_industries = sector_dicts['I']

        df = df_e_slim.copy()
        df['type'] = np.select(
            [df['branche_code'].isin(c_industries),
             df['branche_code'].isin(i_industries)],
            ['C', 'I'],
            default='x'
        )

        weighted_avg = (
            df.groupby(['TID', 'type'], group_keys=False)
            .apply(lambda x: (x['labor_share'] * x['GVA']).sum() / x['GVA'].sum(),
                   include_groups=False)
            .reset_index(name='weighted_labor_share')
        )
        weighted_avg = weighted_avg[weighted_avg['type'].isin(['C', 'I'])]

        pivoted = weighted_avg.pivot(index='TID', columns='type', values='weighted_labor_share').reset_index()
        pivoted['diff'] = pivoted['I'] - pivoted['C']
        pivoted['variation'] = label
        results[label] = pivoted

        ax.plot(pivoted['TID'], pivoted['diff'], color=color, lw=2, ls='--', label=label)

    ax.axhline(0, color="0.2", linewidth=1.2, ls='--')
    ax.set_xlim(pivoted['TID'].min(), pivoted['TID'].max())
    ax.set_ylabel(r'$\Delta(I − C.)\; pp.$')
    ax.grid(linewidth=0.6, alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.legend(loc='lower left')
    fig.suptitle('Labour share difference: investment vs consumption sector', y=0.95)
    plt.tight_layout()

    plt.savefig('0_output/LS_2.png', dpi=200)
    plt.show()
    return results