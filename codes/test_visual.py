import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Assuming all your CSV files are in the same directory and have the pattern 'results*.csv'
file_pattern = 'CPLP_test1/test_results*.csv'
csv_files = glob.glob(file_pattern)

# Create an empty DataFrame to hold all Gap(%) data
gap_data = pd.DataFrame()

# Loop through all files and read them into DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    # Extract the file identifier from the filename, e.g., 'results_10_10_0.csv' -> '10_10_0'
    file_id = file.split('/')[-1].replace('test_results_', '').rstrip('.csv')
    # Add the 'Gap(%)' column to the gap_data DataFrame with a column name as the file identifier
    gap_data[file_id] = df['Gap(%)']
   
# Now, gap_data has each file's Gap(%) as a separate column

# Plotting the boxplot

plot_filename = f'boxplot_gap_percent.png'
plt.figure(figsize=(12, 6))
gap_data.boxplot()
plt.title('Gap(%) Boxplot for Each Data File')
plt.ylabel('Gap(%)')
plt.xlabel('Data File Identifier')
plt.xticks(rotation=45)  # Rotate x labels if they overlap
plt.tight_layout()  # Adjust layout for better fit
plot_save_path = os.path.join("CPLP_test1/", plot_filename)
plt.savefig(plot_save_path)
plt.close()
