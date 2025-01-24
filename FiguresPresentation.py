import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Define the directory containing the CSV files
directory = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs'

# Find all CSV files matching the pattern "XXXXX_Kd.csv"
csv_files = glob.glob(os.path.join(directory,'**', '*_Kd.csv'),recursive= True)

# Initialize lists to store longitude, latitude values, and quality
lons_good = []
lats_good = []
lons_bad = []
lats_bad = []

# Loop through each CSV file and extract lon, lat, and quality columns
for file in csv_files:
    df = pd.read_csv(file)
    if 'lon' in df.columns and 'lat' in df.columns and 'quality' in df.columns:
        for _, row in df.iterrows():
            if row['quality'] == 0:
                lons_good.append(row['lon'])
                lats_good.append(row['lat'])
            elif row['quality'] == 1:
                lons_bad.append(row['lon'])
                lats_bad.append(row['lat'])

# Plot the world map with Robinson projection
plt.figure(figsize=(12, 8))
m = Basemap(projection='robin', lon_0=0)
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='lightgrey', lake_color='white')
m.drawparallels(range(-90, 91, 30), labels=[1,0,0,0])
m.drawmeridians(range(-180, 181, 60), labels=[0,0,0,1])


# Convert latitude and longitude to map projection coordinates
x_good, y_good = m(lons_good, lats_good)
x_bad, y_bad = m(lons_bad, lats_bad)

# Plot the data points
m.scatter(x_bad, y_bad, color='orange', s=12, alpha=0.6, label='Quality 1')

m.scatter(x_good, y_good, color='green', s=10, alpha=0.6, label='Quality 0')

# Add title
plt.title('Green: Good Kd spectra, Orange: Bad/Questionnable Kd spectra',fontsize=20)


plt.savefig('Kd_Spectra_Map.png')
# Show the plot
plt.show()
# %%

# Define the directory containing the CSV files
directory = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs'

# Find all CSV files matching the pattern "XXXXX_Kd.csv"
csv_files = glob.glob(os.path.join(directory, '**', '*_Kd.csv'), recursive=True)

# Initialize lists to store dates for good and bad profiles
dates_good = []
dates_bad = []

# Loop through each CSV file and extract the date and quality columns
for file in csv_files:
    df = pd.read_csv(file)
    if 'date' in df.columns and 'quality' in df.columns:
        # Convert the date column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # Filter dates from January 2024 onwards
        df = df[df['date'] >= '2024-03-05']
        # Append the dates to the respective lists based on quality
        dates_good.extend(df[df['quality'] == 0]['date'])
        dates_bad.extend(df[df['quality'].isin([1, 2])]['date'])

# Create DataFrames from the dates lists
dates_good_df = pd.DataFrame(dates_good, columns=['date'])
dates_bad_df = pd.DataFrame(dates_bad, columns=['date'])

# Extract year and month from the date
dates_good_df['year_month'] = dates_good_df['date'].dt.to_period('M')
dates_bad_df['year_month'] = dates_bad_df['date'].dt.to_period('M')

# Count the number of data points for each month
date_counts_good = dates_good_df['year_month'].value_counts().sort_index()
date_counts_bad = dates_bad_df['year_month'].value_counts().sort_index()


# Calculate the overall sum of good and bad profiles
total_good = date_counts_good.sum()
total_bad = date_counts_bad.sum()

# Align the indices of the two series
date_counts_good = date_counts_good.reindex(date_counts_bad.index, fill_value=0)
date_counts_bad = date_counts_bad.reindex(date_counts_good.index, fill_value=0)

# Plot the stacked bar chart
plt.figure(figsize=(10, 6))
ax = date_counts_good.plot(kind='bar', color='green', label='Good Kd Spectra')
date_counts_bad.plot(kind='bar', color='orange', bottom=date_counts_good, label='Bad/Questionable Kd Spectra', ax=ax)
plt.xlabel('Month')
plt.ylabel('Number of Kd spectra acquired')
plt.title('Number of Kd spectra per Month since January 2024')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Add a vertical red line on February 2024
feb_2024_index = date_counts_good.index.get_loc('2024-08')
ax.axvline(x=feb_2024_index, color='red', linestyle='--', linewidth=2)

plt.savefig('Data_Points_Per_Month_2024.png')

# Show the plot
plt.show()