import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Define the file path
file_path = '/Users/charlotte.begouen/Downloads/QC_fleet_allspectra.csv'
df = pd.read_csv(file_path)
#qc_mapping = {1: 2, 2: 3, 3: 1}

QC_5 = pd.read_csv('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs/QC_5wv.csv')
QC_5['WMO'] = [f"{profile}_{str(cycle_number).zfill(3)}" for profile, cycle_number in zip(QC_5['wmo'], QC_5['cycle_number'])]
QC_5['quality_5wv'] = QC_5['quality_5wv'] + 1
# remove column wmo and cycle_number
QC_5 = QC_5.drop(columns=['wmo', 'cycle_number'])

# Import all Charlotte's Kd files
directory = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs'
# Find all CSV files matching the pattern "*_Kd.csv"
csv_files = glob.glob(os.path.join(directory, '**', '*_Kd.csv'), recursive=True)

# Initialize an empty list to store the data
data = []

# Loop through each CSV file and extract the required information
for file in csv_files:
    # Extract the WMO from the filename
    wmo = os.path.basename(file).split('_Kd.csv')[0]

    kd_file = pd.read_csv(file)
    # Create an empty file
    # Create a new DataFrame with the required columns
    temp_df = pd.DataFrame()
    temp_df['WMO'] = [f"{wmo}_{str(profile).zfill(3)}" for profile in kd_file['profile']]
    temp_df['quality'] = kd_file['quality']

    # Append the data to the list
    data.append(temp_df.values)

# Concatenate all the data into a single DataFrame
result_df = pd.DataFrame(np.concatenate(data), columns=['WMO', 'quality'])
result_df['quality'] = result_df['quality'] + 1
# Save to a CSV file
result_df.to_csv('/Users/charlotte.begouen/Downloads/Charlotte_QC.csv', index=False)

# Merge df and result_df based on the WMO column
merged_df = pd.merge(df, result_df, on='WMO', how='left')
merged_df = pd.merge(merged_df, QC_5, on='WMO', how='left')

merged_df.to_csv('/Users/charlotte.begouen/Downloads/3QC_Comparison_20241203.csv', index=False)
#
# # Identify the QC columns
# qc_columns = [col for col in df.columns if 'QC' in col]
# # Replace the values in the QC columns
# df[qc_columns] = df[qc_columns].replace(qc_mapping)
# Create a list to store the dictionaries
percentages_list = []

# Calculate the percentage of data for each combination
total_rows = len(merged_df)
for type_val in [1, 2, 3]:
    for qc_val in [1, 2, 3]:
        for quality_5wv_val in [1, 2, 3]:
            count = len(merged_df[(merged_df['TYPE'] == type_val) & (merged_df['quality'] == qc_val) & (merged_df['quality_5wv'] == quality_5wv_val)])
            percentage = (count / total_rows) * 100
            percentages_list.append({'Lou_QC': type_val, 'Charlotte_QC': qc_val, 'Quality_5wv': quality_5wv_val, 'Percentage': percentage})

# Convert the list to a DataFrame
percentages = pd.DataFrame(percentages_list)
percentages.to_csv('/Users/charlotte.begouen/Downloads/Comparison_3wv_percentages.csv', index=False)

# Create a list to store the dictionaries
percentages_list = []

# Calculate the percentage of data for each combination
total_rows = len(merged_df)
for type_val in [1, 2, 3]:
    for quality_5wv_val in [1, 2, 3]:
        count = len(merged_df[(merged_df['TYPE'] == type_val) & (merged_df['quality'] == quality_5wv_val)])
        percentage = (count / total_rows) * 100
        percentages_list.append({'Lou_QC': type_val, 'Quality': quality_5wv_val, 'Percentage': percentage})

# Convert the list to a DataFrame
percentages = pd.DataFrame(percentages_list)
percentages.to_csv('/Users/charlotte.begouen/Downloads/Comparison_LouCha_percentages.csv', index=False)


# Create a pivot table for the heatmap
pivot_table = percentages.pivot_table("Percentage", "Quality", "Lou_QC")

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
plt.title('Percentage of Data for Each Combination of TYPE_after_range and QC')
plt.xlabel('Charlotte_QC')
plt.ylabel('Lou_QC')
plt.show()

# Save the DataFrame as odf.csv
df.to_csv('/Users/charlotte.begouen/Downloads/CompareQC_corrected.csv', index=False)

# %%

# Loop through each CSV file and extract the required information
for file in csv_files:
    # Extract the WMO from the filename
    wmo = os.path.basename(file).split('_Kd.csv')[0]

    kd_file = pd.read_csv(file)
    # Create an empty file
    # Create a new DataFrame with the required columns
    temp_df = pd.DataFrame()
    temp_df['WMO'] = [f"{wmo}_{str(profile).zfill(3)}" for profile in kd_file['profile']]
    temp_df['lon'] = kd_file['lon']
    temp_df['lat'] = kd_file['lat']

    # Append the data to the list
    data.append(temp_df.values)

# Concatenate all the data into a single DataFrame
result_df = pd.DataFrame(np.concatenate(data), columns=['WMO', 'lon', 'lat'])
# Save to a CSV file
result_df.to_csv('/Users/charlotte.begouen/Downloads/Location_hyperspectral.csv', index=False)
