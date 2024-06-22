# import pandas as pd
# import matplotlib.pyplot as plt
# import glob
# import os

# # Assuming all your CSV files are in the same directory and have the pattern 'results*.csv'
# file_pattern = 'CPLP_test1/test_results*.csv'
# csv_files = glob.glob(file_pattern)

# # Create an empty DataFrame to hold all Gap(%) data
# gap_data = pd.DataFrame()

# # Loop through all files and read them into DataFrames
# for file in csv_files:
#     df = pd.read_csv(file)
#     # Extract the file identifier from the filename, e.g., 'results_10_10_0.csv' -> '10_10_0'
#     file_id = file.split('/')[-1].replace('test_results_', '').rstrip('.csv')
#     # Add the 'Gap(%)' column to the gap_data DataFrame with a column name as the file identifier
#     gap_data[file_id] = df['Gap(%)']
   
# # Now, gap_data has each file's Gap(%) as a separate column

# # Plotting the boxplot

# plot_filename = f'boxplot_gap_percent.png'
# plt.figure(figsize=(12, 6))
# gap_data.boxplot()
# plt.title('Gap(%) Boxplot for Each Problem Size')
# plt.ylabel('Gap(%)')
# plt.xlabel('Problem Size (client_plant)')
# plt.xticks(rotation=45)  # Rotate x labels if they overlap
# plt.tight_layout()  # Adjust layout for better fit
# plot_save_path = os.path.join("CPLP_test1/", plot_filename)
# plt.savefig(plot_save_path)
# plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Assuming all your CSV files are in the same directory and have the pattern 'test_results*.csv'
file_pattern = 'CPLP_test1/test_results*.csv'
csv_files = glob.glob(file_pattern)

# Create an empty dictionary to hold data grouped by problem size
gap_data_by_size = {}

# Loop through all files and read them into DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    # Extract the problem size information from the file name
    size_info = file.split('/')[-1].split('_')[2:4]  # Get the size parts from the filename
    size_key = f"{size_info[0]}_{size_info[1]}"  # Create a key for sizes
    
    # If this size_key is not in the dictionary, add it with an empty list
    if size_key not in gap_data_by_size:
        gap_data_by_size[size_key] = []

    # Append the 'Gap(%)' data to the appropriate size_key's list
    gap_data_by_size[size_key].extend(df['Gap(%)'].tolist())

# Prepare the boxplot data (a list of arrays, one for each problem size)
boxplot_data = [gaps for gaps in gap_data_by_size.values()]
size_labels = [size for size in gap_data_by_size.keys()]

# Plotting the combined boxplot for all problem sizes
plt.figure(figsize=(12, 6))
plt.boxplot(boxplot_data, labels=size_labels)
plt.title('Optimality Gap by Problem Size', fontsize=15)
plt.ylabel('Gap(%)', fontsize=15)
plt.xlabel('Problem Size ("Number of client"_"Num of plant")', fontsize=15)
plt.xticks(rotation=0)  # Rotate x labels if they overlap
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.tight_layout()  # Adjust layout for better fit
plt.savefig(os.path.join("CPLP_test1", "boxplot_gap_percent_by_size.png"))
plt.show()

