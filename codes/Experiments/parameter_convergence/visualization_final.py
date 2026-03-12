import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

mpl.rcParams['axes.facecolor'] = '#fafafa'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['grid.color'] = '0.6'
mpl.rcParams['grid.alpha'] = 0.30
mpl.rcParams['axes.edgecolor'] = '0.6'
mpl.rcParams['legend.framealpha'] = 0.95
mpl.rcParams['legend.edgecolor'] = '0.6'

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern'
mpl.rcParams['font.weight'] = 'heavy'
    
palette = sns.color_palette("colorblind")


def load_data():
    try:
        fixed_summary = pd.read_csv('aggregated_results/fixed_summary.csv')
        adaptive_summary = pd.read_csv('aggregated_results/adaptive_summary.csv')
        return fixed_summary, adaptive_summary
    except FileNotFoundError as e:
        print(f"error: {e}")
        return None, None


def floor_dec(x, decimals=2):
    factor = 10 ** decimals
    return math.floor(x * factor + 1e-9) / factor


def create_gap_heatmap(ax, fixed_summary, adaptive_summary):
    ref_row = fixed_summary.loc[
        (fixed_summary['L'] == fixed_summary['L'].max()) &
        (fixed_summary['N'] == fixed_summary['N'].max())
    ]
    ref_cost = ref_row['mean_E_Q'].values[0]

    fixed_summary = fixed_summary.copy()
    fixed_summary['rel_error'] = (
        (fixed_summary['mean_E_Q'] - ref_cost) / ref_cost * 100
    ).abs()

    pivot_rel = fixed_summary.pivot(
        index='L', columns='N', values='rel_error'
    ).sort_index(ascending=False)

    try:
        annot_data = pivot_rel.applymap(lambda x: f'{floor_dec(x, 2):.2f}')
    except AttributeError:
        annot_data = pivot_rel.map(lambda x: f'{floor_dec(x, 2):.2f}')

    sns.heatmap(
        pivot_rel, annot=annot_data, fmt='',
        cmap='RdYlGn_r', linewidths=2.0, linecolor='white',
        cbar=False, annot_kws={'size': 22, 'weight': 'bold'}, ax=ax
    )

    ax.set_title(
        r'Relative Error in $\widehat{\mathcal{Q}}$ (\%)',
        fontsize=24, fontweight='bold', pad=20
    )

    avg_L = math.floor(adaptive_summary['avg_L'].mean())
    avg_N = math.floor(adaptive_summary['avg_N'].mean())

    L_values = sorted(list(pivot_rel.index), reverse=True)
    N_values = sorted(list(pivot_rel.columns))

    y_continuous = np.interp(avg_L, L_values[::-1], range(len(L_values))[::-1])
    x_continuous = np.interp(avg_N, N_values, range(len(N_values)))

    adaptive_cost = adaptive_summary['total_E_Q'].mean()
    adaptive_rel_error = abs((adaptive_cost - ref_cost) / ref_cost * 100)

    ax.scatter(
        x_continuous + 0.5, y_continuous + 0.5,
        marker='*', s=1000, c='darkorange', linewidths=3,
        label=(f'Adaptive ($|\\mathcal{{H}}|$={avg_L}, S={avg_N},'
               f' {adaptive_rel_error:.2f}\\%)'),
        zorder=8
    )

    ax.set_xlabel(r'Number of Scenarios (S)', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'Operational Horizon ($|\mathcal{H}|$)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='upper right', fontsize=18)


def plot_time_bar_colored(ax, fixed_df, adaptive_df):
    ref_row = fixed_df.loc[
        (fixed_df['L'] == fixed_df['L'].max()) &
        (fixed_df['N'] == fixed_df['N'].max())
    ]
    ref_cost = ref_row['mean_E_Q'].values[0]

    selected_configs = [
        (5, 12), (10, 24), (10, 36), (20, 36), (30, 48),
    ]

    labels, times, errors = [], [], []

    for (S, H) in selected_configs:
        row = fixed_df[(fixed_df['N'] == S) & (fixed_df['L'] == H)]
        if len(row) > 0:
            labels.append(f'($S$={S}, $|\\mathcal{{H}}|$={H})')
            times.append(row['avg_exec_time'].values[0])
            cost = row['mean_E_Q'].values[0]
            errors.append(abs((cost - ref_cost) / ref_cost * 100))

    adaptive_avg_time = adaptive_df['avg_exec_time'].mean()
    adaptive_avg_L = math.floor(adaptive_df['avg_L'].mean())
    adaptive_avg_N = math.floor(adaptive_df['avg_N'].mean())
    adaptive_cost = adaptive_df['total_E_Q'].mean()
    adaptive_error = abs((adaptive_cost - ref_cost) / ref_cost * 100)

    labels.append(f'Adaptive\n($S$={adaptive_avg_N}, $|\\mathcal{{H}}|$={adaptive_avg_L})')
    times.append(adaptive_avg_time)
    errors.append(adaptive_error)

    x_pos = np.arange(len(labels))

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=max(errors) * 1.1)
    bar_colors = [cmap(norm(e)) for e in errors]

    bars = ax.bar(x_pos, times, color=bar_colors, width=0.6,
                  edgecolor='0.3', linewidth=1.0)

    bars[-1].set_edgecolor('darkorange')
    bars[-1].set_linewidth(3.0)

    for bar, t in zip(bars, times):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, bar.get_height() + 3, f'{t:.0f}s',
                ha='center', va='bottom', fontsize=20, fontweight='bold')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label(r'Relative Error (\%)', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)

    max_fixed_time = max(times[:-1])
    ax.axhline(y=max_fixed_time, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=15)
    ax.set_ylabel('Time per Label (seconds)', fontsize=24, fontweight='bold')
    ax.set_title('Wall-Clock Time Comparison', fontsize=24, fontweight='bold', pad=16)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def main():
    fixed_summary, adaptive_summary = load_data()
    if fixed_summary is None:
        print("No data found. Exiting.")
        return

    fig = plt.figure(figsize=(28, 9.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1], wspace=0.18)

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    create_gap_heatmap(ax_heatmap, fixed_summary, adaptive_summary)
    plot_time_bar_colored(ax_bar, fixed_summary, adaptive_summary)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()