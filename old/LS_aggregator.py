import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_labor_share_variations(df_e_slim, variations: dict, colors=None):
    if colors is None:
        colors = ['#F76A4D', '#41FAB4', '#4D9FF7', '#F7D94D', '#A44DF7']

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax_twin = ax.twinx()   
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

        pivoted = (
            weighted_avg
            .pivot(index='TID', columns='type', values='weighted_labor_share')
            .reset_index()
        )
        pivoted['diff'] = pivoted['I'] - pivoted['C']
        pivoted['variation'] = label
        results[label] = pivoted

        ax.plot(
            pivoted['TID'],
            pivoted['diff'], 
            color=color,
            lw=2,
            ls='-',
            label=label
        )

    # plot right-axis series only once
    capinc = (
        df_e_slim.groupby('TID', as_index=False)['capinc_share']
        .mean()
    )
    # ax_twin.plot(
    #     capinc['TID'],
    #     capinc['capinc_share'] * 100, 
    #     color='#4A4A5A',
    #     lw=2,
    #     ls='--',
    #     label='Avg. capital/wage income ratio'
    # )

    ax.axhline(0, color="0.2", linewidth=1.2, ls='--')
    ax.set_xlim(capinc['TID'].min(), capinc['TID'].max())
    ax.set_ylabel(r'$\Delta LS \;(I-C)\;$ pp.')
    ax_twin.set_ylabel('Capital/wage ratio')

    ax.grid(linewidth=0.6, alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))

    # percent formatting on right axis
    # ax_twin.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    # combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc='lower left')

    fig.suptitle('Labour share difference: investment vs consumption sector', y=0.95)
    plt.tight_layout()

    plt.savefig('0_output/LS_2.png', dpi=200)
    plt.show()
    return results
