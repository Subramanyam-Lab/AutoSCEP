import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os
from scipy import stats

# module load texlive/2024...

try:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
except Exception as e:
    mpl.rcParams['text.usetex'] = False

mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300



def plot_investment_evolution(ef_file_path, ml_folder_path, save_path=None):
    ef_data = pd.read_csv(ef_file_path)
    ef_data['Method'] = 'EF'
    ef_data['Run'] = 0

    file_paths = glob.glob(os.path.join(ml_folder_path, 'ML_Embed_installed_solution_*.csv'))
    ml_data_list = []
    for path in file_paths:
        filename = os.path.basename(path).replace('.csv', '')
        parts = filename.split('_')
        model, numsce, run = parts[4], parts[5], parts[6]
        
        temp_df = pd.read_csv(path)
        temp_df['Method'] = f'{model}_{numsce}'
        temp_df['Run'] = int(run)
        ml_data_list.append(temp_df)
    
    all_ml_data = pd.concat(ml_data_list, ignore_index=True)
    combined_data = pd.concat([ef_data, all_ml_data], ignore_index=True)
    combined_data['Year'] = 2020 + (combined_data['Period'] - 1) * 5
    
    colors = {
        'EF': '#000000', 'MLP_1000': '#E63946', 'MLP_5000': '#F77F00',
        'LR_1000': '#06AED5', 'LR_5000': '#70C1B3'
    }
    specific_types = ['Generation', 'Transmission', 'Storage Power']
    methods = ['EF', 'MLP_1000', 'MLP_5000', 'LR_1000', 'LR_5000']
    
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    axes = axes.flatten()
    
    for ax_idx, inv_type in enumerate(specific_types):
        ax = axes[ax_idx]
        type_data = combined_data[combined_data['Type'] == inv_type]
        sum_data = type_data.groupby(['Year', 'Method', 'Run'])['Value'].sum().reset_index()
        
        for method in methods:
            method_data = sum_data[sum_data['Method'] == method]
            
            if method == 'EF':
                ef_values = method_data.groupby('Year')['Value'].mean().reset_index()
                ax.plot(ef_values['Year'], ef_values['Value']/1000,
                       color=colors[method], linewidth=2, marker='o', markersize=8,
                       label='EF(100)', zorder=10)
            else:
                stats_data = method_data.groupby('Year')['Value'].agg(['mean', 'std', 'count']).reset_index()
                confidence = 0.95
                stats_data['ci'] = stats_data['std'] / np.sqrt(stats_data['count']) * \
                                   stats.t.ppf((1 + confidence) / 2, stats_data['count'] - 1)
                
                label = 'A-' + method.replace('_', '(').replace('1000', '1K').replace('5000', '5K') + ')'
                ax.plot(stats_data['Year'], stats_data['mean']/1000,
                       color=colors[method], linewidth=2, marker='s', markersize=6,
                       label=label, alpha=0.9, linestyle='--')
                
                ax.fill_between(stats_data['Year'],
                               (stats_data['mean'] - stats_data['ci'])/1000,
                               (stats_data['mean'] + stats_data['ci'])/1000,
                               color=colors[method], alpha=0.15)
        
        ax.set_title(f'Installed Capacity for {inv_type}', fontsize=22)
        ax.set_xlabel('Year', fontsize=20)
        ax.set_ylabel('Capacity (GW)', fontsize=20)
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return fig


def main():
    ef_file_path = '100_seed_5_installed_cap.csv' # this is EF(100) solution for EMPIRE-sml
    ml_folder_path = '../MLEMBEDSOLS_adaptive/'
    
    print("Creating visualizations...")
    print("Generating: Investment Evolution...")
    plot_investment_evolution(
        ef_file_path, ml_folder_path,
        save_path='aggregated_investment_capacity_comparison.png'
    )
        
    print("\nAll visualizations completed and saved!")

if __name__ == "__main__":
    main()
