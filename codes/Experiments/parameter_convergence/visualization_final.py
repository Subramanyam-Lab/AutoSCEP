import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

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
fixed_method_color = palette[0]
our_method_color = palette[1]
savings_fill_color = 'gray'

def load_data():
    try:
        fixed_summary = pd.read_csv('aggregated_results/fixed_summary.csv')
        adaptive_summary = pd.read_csv('aggregated_results/adaptive_summary.csv')
        return fixed_summary, adaptive_summary
    except FileNotFoundError as e:
        print(f"error: {e}")
        fixed_summary = pd.DataFrame({
            'L': np.repeat(np.arange(10, 51, 10), 4),
            'N': np.tile(np.arange(10, 41, 10), 5),
            'mean_E_Q': np.random.rand(20) * 1e6,
            'avg_exec_time': np.random.rand(20) * 100 + 50
        })
        adaptive_summary = pd.DataFrame({
            'avg_L': [35.5], # random number 
            'avg_N': [15.2], # random number 
            'total_E_Q': [np.random.rand() * 1e6],
            'avg_exec_time': [42.5]
        })
        return fixed_summary, adaptive_summary


def create_gap_heatmap(ax, fixed_summary, adaptive_summary):
    pivot_eq = fixed_summary.pivot(
        index='L', 
        columns='N', 
        values='mean_E_Q'
    ).sort_index(ascending=False)
    
    cmap = 'RdYlGn_r'
    
    scale_factor = 1e12  
    pivot_eq_scaled = pivot_eq / scale_factor
    annot_data = pivot_eq_scaled.applymap(lambda x: f'{x:.2f}')
    
    sns.heatmap(
        pivot_eq, 
        annot=annot_data,
        fmt='',
        cmap=cmap, 
        linewidths=2.0, 
        linecolor='white',
        cbar=False,
        annot_kws={'size': 22, 'weight': 'bold'},
        ax=ax
    )
    
    ax.set_title(r'Estimated Expected Production Cost $\widehat{\mathcal{Q}}$', fontsize=24, fontweight='bold', pad=20)
    avg_L = adaptive_summary['avg_L'].mean()
    avg_N = adaptive_summary['avg_N'].mean()
    
    L_values = sorted(list(pivot_eq.index), reverse=True)
    N_values = sorted(list(pivot_eq.columns))
    
    y_continuous = np.interp(avg_L, L_values[::-1], range(len(L_values))[::-1])
    x_continuous = np.interp(avg_N, N_values, range(len(N_values)))
    
    ax.scatter(
        x_continuous + 0.5, y_continuous + 0.5,
        marker='*', s=1000, c='darkorange',
        linewidths=3,
        label=f'Adaptive ($|\mathcal{{H}}|$={avg_L:.0f}, S={avg_N:.0f})',
        zorder=8
    )
    
    ax.set_xlabel(r'Number of Scenarios (S)', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'Operational Horizon ($|\mathcal{H}|$)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='upper right', fontsize=24)

def plot_time_vs_h(ax, fixed_df, adaptive_df):
    S_fixed = 10
    fixed_S10 = fixed_df[fixed_df['N'] == S_fixed].sort_values('L')
    adaptive_avg_time = 58.9817
    
    ax.plot(fixed_S10['L'], fixed_S10['avg_exec_time'], 
             marker='o', linewidth=2.5, markersize=8, 
             label='Fixed Parameters', color=fixed_method_color)

    ax.axhline(y=adaptive_avg_time, color=our_method_color, linestyle='--',
                linewidth=2.5, label='Adaptive Parameter Selection')

    ax.fill_between(fixed_S10['L'], fixed_S10['avg_exec_time'], adaptive_avg_time,
                      alpha=0.2, color=savings_fill_color, label='Computational Savings')

    ax.set_xlabel(r'Operational Horizon ($|\mathcal{H}|$)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Time per Label (seconds)', fontsize=24, fontweight='bold')
    ax.set_title(f'$S = {S_fixed}$ (fixed)', fontsize=24, fontweight='bold')
    ax.legend(loc='best', fontsize=24)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=24)

def plot_time_vs_s(ax, fixed_df, adaptive_df):
    H_fixed = 36
    available_H = fixed_df['L'].unique()
    H_fixed = min(available_H, key=lambda x:abs(x-H_fixed))
    
    fixed_H_filtered = fixed_df[fixed_df['L'] == H_fixed].sort_values('N')
    adaptive_avg_time = adaptive_df['avg_exec_time'].mean()
    
    ax.plot(fixed_H_filtered['N'], fixed_H_filtered['avg_exec_time'], 
             marker='s', linewidth=2.5, markersize=8, 
             label='Fixed Parameters', color=fixed_method_color)

    ax.axhline(y=adaptive_avg_time, color=our_method_color, linestyle='--',
                linewidth=2.5, label='Adaptive Parameter Selection')

    ax.fill_between(fixed_H_filtered['N'], fixed_H_filtered['avg_exec_time'], adaptive_avg_time,
                      alpha=0.2, color=savings_fill_color, label='Computational Savings')

    ax.set_xlabel(r'Number of Scenarios ($S$)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Time per Label (seconds)', fontsize=24, fontweight='bold')
    ax.set_title(f'$|\mathcal{{H}}| = {H_fixed}$ (fixed)', fontsize=24, fontweight='bold')
    ax.legend(loc='best', fontsize=24)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlim(left=0, right=fixed_H_filtered['N'].max() + 5)


def main():
    fixed_summary, adaptive_summary = load_data()
    
    fig = plt.figure(figsize=(27, 9.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.6])

    ax_heatmap = fig.add_subplot(gs[:, 0])  
    ax_time_h = fig.add_subplot(gs[0, 1])   
    ax_time_s = fig.add_subplot(gs[1, 1])   
    
    create_gap_heatmap(ax_heatmap, fixed_summary, adaptive_summary)
    plot_time_vs_h(ax_time_h, fixed_summary, adaptive_summary)
    plot_time_vs_s(ax_time_s, fixed_summary, adaptive_summary)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
    
    
if __name__ == "__main__":
    main()