#!/usr/bin/env python
# aggregate_results.py
import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def aggregate_fixed_results():
    input_dir = "sampling_convergence_controlled"
    
    if not os.path.exists(input_dir):
        logging.error(f"Directory {input_dir} does not exist!")
        return None
    
    fixed_files = glob.glob(os.path.join(input_dir, "fixed_*.csv"))
    
    if not fixed_files:
        logging.error("No fixed files found!")
        return None
    
    logging.info(f"Found {len(fixed_files)} fixed files")    
    data_by_seed = defaultdict(lambda: defaultdict(list))
    
    for filepath in fixed_files:
        try:
            filename = os.path.basename(filepath)
            parts = filename.replace('fixed_', '').replace('.csv', '').split('_')
            
            L = int(parts[0])
            N = int(parts[1])
            period = int(parts[2])
            mseed = int(parts[3].replace('mseed', ''))
            
            df = pd.read_csv(filepath)
            
            if len(df) > 0:
                row = df.iloc[0]
                E_Q = float(row['E_Q_i'])
                c_i = float(row['c_i'])
                exec_time = float(row['execution_time'])
                
                data_by_seed[mseed][(L, N)].append({
                    'period': period,
                    'E_Q_i': E_Q,
                    'c_i': c_i,
                    'exec_time': exec_time
                })
                
        except Exception as e:
            logging.warning(f"Error processing {filepath}: {e}")
            continue
    
    aggregated_by_seed = {}
    
    for mseed, data in data_by_seed.items():
        results = []
        for (L, N), periods_data in data.items():
            total_E_Q = sum(p['E_Q_i'] for p in periods_data)
            avg_c_i = np.mean([p['c_i'] for p in periods_data])
            total_exec_time = sum(p['exec_time'] for p in periods_data)
            num_periods = len(periods_data)
            
            results.append({
                'master_seed': mseed,
                'L': L,
                'N': N,
                'total_E_Q': total_E_Q,
                'avg_c_i': avg_c_i,
                'total_exec_time': total_exec_time,
                'num_periods': num_periods
            })
        
        aggregated_by_seed[mseed] = pd.DataFrame(results)
        logging.info(f"Master seed {mseed}: {len(results)} (L, N) combinations")
    
    all_results = pd.concat(aggregated_by_seed.values(), ignore_index=True)
    
    summary = all_results.groupby(['L', 'N']).agg({
        'total_E_Q': ['mean', 'std', 'min', 'max'],
        'total_exec_time': 'mean',
        'num_periods': 'first'
    }).reset_index()
    
    summary.columns = ['L', 'N', 'mean_E_Q', 'std_E_Q', 'min_E_Q', 'max_E_Q', 'avg_exec_time', 'num_periods']    
    summary['CV'] = (summary['std_E_Q'] / summary['mean_E_Q']) * 100
    
    return {
        'by_seed': aggregated_by_seed,
        'all_data': all_results,
        'summary': summary
    }


def aggregate_adaptive_results():
    input_dir = "sampling_convergence_controlled"
    adaptive_files = glob.glob(os.path.join(input_dir, "adaptive_*.csv"))
    
    if not adaptive_files:
        logging.error("No adaptive files found!")
        return None
    
    logging.info(f"Found {len(adaptive_files)} adaptive files")
    data_by_seed = defaultdict(list)
    
    for filepath in adaptive_files:
        try:
            filename = os.path.basename(filepath)
            parts = filename.replace('adaptive_', '').replace('.csv', '').split('_')
            
            period = int(parts[0])
            mseed = int(parts[1].replace('mseed', ''))
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                continue
            
            header = lines[0].strip().split(',')
            
            if 'file_num' in header:
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    row = df.iloc[0]
                    L = int(row['lengthRegSeason'])
                    N = int(row['N_i'])
                    E_Q = float(row['E_Q_i'])
                    c_i = float(row['c_i'])
                    status = row['status']
                    exec_time = float(row['execution_time'])
            else:
                data_line = lines[1].strip()
                parts = data_line.split(',')
                
                numeric_parts = []
                for p in parts:
                    p = p.strip().strip("'\"")
                    try:
                        val = float(p)
                        numeric_parts.append(val)
                    except:
                        pass
                
                if len(numeric_parts) >= 5:
                    E_Q = numeric_parts[-4]
                    N = int(numeric_parts[-3])
                    L = int(numeric_parts[-2])
                    exec_time = numeric_parts[-1] if len(numeric_parts) > 4 else 0
                else:
                    continue
                
                status = "Unknown"
                for p in parts:
                    if any(s in p for s in ['Converged', 'Terminated']):
                        status = p.strip().strip("'\"")
                        break
                
                c_i = numeric_parts[-5] if len(numeric_parts) > 5 else 0
            
            data_by_seed[mseed].append({
                'period': period,
                'L': L,
                'N': N,
                'E_Q_i': E_Q,
                'c_i': c_i,
                'status': status,
                'exec_time': exec_time
            })
            
        except Exception as e:
            logging.warning(f"Error processing {filepath}: {e}")
            continue
        
    aggregated_by_seed = {}
    
    for mseed, periods_data in data_by_seed.items():
        df = pd.DataFrame(periods_data)
        exec_times = df['exec_time'].tolist()
        total_E_Q = df['E_Q_i'].sum()
        avg_L = df['L'].mean()
        avg_N = df['N'].mean()
        avg_c_i = df['c_i'].mean()
        
        total_exec_time = df['exec_time'].sum()
        avg_exec_time = df['exec_time'].mean()
        
        num_periods = len(df)
        
        aggregated_by_seed[mseed] = {
            'master_seed': mseed,
            'total_E_Q': total_E_Q,
            'avg_L': avg_L,
            'avg_N': avg_N,
            'avg_c_i': avg_c_i,
            'total_exec_time': total_exec_time,  
            'avg_exec_time': avg_exec_time,      
            'num_periods': num_periods,
            'periods_data': df
        }
        
        logging.info(f"Master seed {mseed}: Avg L={avg_L:.1f}, Avg N={avg_N:.1f}, "
                    f"Total Time={total_exec_time:.2f}s)")
    
    summary_data = []
    for mseed, data in aggregated_by_seed.items():
        summary_data.append({
            'master_seed': mseed,
            'total_E_Q': data['total_E_Q'],
            'avg_L': data['avg_L'],
            'avg_N': data['avg_N'],
            'total_exec_time': data['total_exec_time'],  
            'avg_exec_time': data['avg_exec_time'],        
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    all_exec_times = []
    for mseed, periods_data in data_by_seed.items():
        df = pd.DataFrame(periods_data)
        all_exec_times.extend(df['exec_time'].tolist())
    
    overall_summary = {
        'mean_E_Q': summary_df['total_E_Q'].mean(),
        'std_E_Q': summary_df['total_E_Q'].std(),
        'mean_L': summary_df['avg_L'].mean(),
        'std_L': summary_df['avg_L'].std(),
        'mean_N': summary_df['avg_N'].mean(),
        'std_N': summary_df['avg_N'].std(),
        'mean_total_exec_time': summary_df['total_exec_time'].mean(),
        'std_total_exec_time': summary_df['total_exec_time'].std(),
        'mean_avg_exec_time': summary_df['avg_exec_time'].mean()
    }
    
    return {
        'by_seed': aggregated_by_seed,
        'summary_df': summary_df,
        'overall_summary': overall_summary
    }


def save_aggregated_data(fixed_results, adaptive_results):
    
    output_dir = "aggregated_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if fixed_results:
        fixed_results['summary'].to_csv(
            os.path.join(output_dir, 'fixed_summary_revision.csv'),
            index=False
        )
        logging.info("Saved: fixed_summary.csv")
        
        for mseed, df in fixed_results['by_seed'].items():
            df.to_csv(
                os.path.join(output_dir, f'fixed_mseed{mseed}.csv'),
                index=False
            )
        logging.info(f"Saved: {len(fixed_results['by_seed'])} fixed seed files")
    
    if adaptive_results:
        adaptive_results['summary_df'].to_csv(
            os.path.join(output_dir, 'adaptive_summary_revision.csv'),
            index=False
        )
        logging.info("Saved: adaptive_summary.csv")

def main():
    logging.info("Starting data aggregation...")
    logging.info("Aggregating Fixed results...")
    fixed_results = aggregate_fixed_results()
    
    logging.info("Aggregating Adaptive results...")
    adaptive_results = aggregate_adaptive_results()
    
    logging.info("Saving aggregated data...")
    save_aggregated_data(fixed_results, adaptive_results)
    
    return fixed_results, adaptive_results


if __name__ == "__main__":
    fixed_results, adaptive_results = main()