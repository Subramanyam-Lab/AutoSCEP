import pandas as pd
import numpy as np
from typing import List, Dict, Union

def sample_fsd_load(dir_seed: str) -> pd.DataFrame:
    try:
        fsd_list = []
        for seed in range(1, 11):
            df = pd.read_csv(f'{dir_seed}/fsd_seed{seed}.csv')
            fsd_list.append(df)
        combined_df = pd.concat(fsd_list)
        fsd_average = combined_df.groupby(['Node', 'Energy_Type', 'Period', 'Type']).mean().reset_index()
        
        # options
        fsd_average.to_csv("fsd_average.csv", index=False)
        return fsd_average
    except Exception as e:
        print(f"Error in sample_fsd_load: {e}")
        return pd.DataFrame()

def compute_minimal(group: pd.Series, threshold: float) -> List[str]:
    cumulative_ratio = group.cumsum()
    sets_within_threshold = cumulative_ratio[cumulative_ratio <= threshold]
    if cumulative_ratio.max() > threshold:
        last_element = cumulative_ratio[cumulative_ratio > threshold].index[0]
        minimal_sets = sets_within_threshold.index.tolist() + [last_element]
    else:
        minimal_sets = sets_within_threshold.index.tolist()
    return minimal_sets

def N_star(df_investments: pd.DataFrame, t: str, epsilon: float) -> Dict[str, List[str]]:
    # P_T
    df_investments = df_investments[df_investments['Type'] == t]
    type_totals = df_investments.groupby('Type')['Value'].sum()
    node_type_totals = df_investments.groupby(['Node', 'Type'])['Value'].sum().unstack(fill_value=0)
    node_type_percentages = node_type_totals.div(type_totals) * 100
    node_type_percentages = node_type_percentages.sort_values(by=t, ascending=False)
    
    # Load N^*_{T} for T \in \mathcal{T}
    minimal_nodes = compute_minimal(node_type_percentages[t], epsilon)
    return {t: minimal_nodes}

def F_star(N_star: List[str], df_investments: pd.DataFrame, t: str, delta: float) -> Dict[str, List[str]]:
    # Filter investments for the given type and nodes
    results_dict = {}
    for n in N_star:
        # df_filtered = df_investments[(df_investments['Type'] == t) & (df_investments['Node'].isin(n))]
        df_filtered = df_investments[(df_investments['Type'] == t) & (df_investments['Node'] == n)]
        # Calculate energy type percentages within the Type
        type_energy_totals = df_filtered.groupby('Energy_Type')['Value'].sum()
        type_energy_percentages = (type_energy_totals / type_energy_totals.sum()) * 100
        type_energy_percentages = type_energy_percentages.sort_values(ascending=False)
        
        # Compute minimal units
        minimal_units = compute_minimal(type_energy_percentages, delta)
        results_dict[n] = {minimal_units}
    return results_dict

def main():
    dir_seed = 'SeedSamples/reduced'
    epsilon = 100
    delta = 80
    results_dict = {}

    try:
        fsd_average = sample_fsd_load(dir_seed)
        if fsd_average.empty:
            raise ValueError("Failed to load data")

        # Filter for the relevant investment types
        investment_types = ['Generation', 'Storage Power', 'Storage Energy']
        df_investments = fsd_average[fsd_average['Type'].isin(investment_types)]

        for t in investment_types:
            N_star_sets = N_star(df_investments, t, epsilon)
            F_star_sets = F_star(N_star_sets[t], df_investments, t, delta)
            results_dict[t] = (F_star_sets)

        print("Analysis completed successfully.")
        print("Results:", results_dict)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()