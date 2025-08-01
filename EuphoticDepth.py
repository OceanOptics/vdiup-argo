import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


par_folder = '/Users/charlotte.begouen/Desktop/Euphotic_study/PAR_files'
kd_folder = '/Users/charlotte.begouen/Desktop/Euphotic_study/Kd_files'
# List all .nc files in the PAR folder
par_files = [f for f in os.listdir(par_folder) if f.endswith('.nc')]
kd_files = [f for f in os.listdir(kd_folder) if f.endswith('.nc')]

for par_file in par_files:
    par_path = os.path.join(par_folder, par_file)
    date_str = par_file.split('.')[1]  # Assuming the date is in the filename
    # Find the corresponding Kd file
    kd_file = next((f for f in kd_files if date_str in f), None)
    kd_path = os.path.join(kd_folder, kd_file)

    ds_par = xr.open_dataset(par_path)
    ds_kd = xr.open_dataset(kd_path)

    #print(ds_par)
    par_data = ds_par['par'].values
    kd_data = ds_kd['Kd_490'].values
    lat= ds_par['lat'].values
    lon = ds_par['lon'].values

    # meshgrid for lat and lon
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Calculate 1% value for PAR
    euphotic_value = 0.01 * par_data

    # Calculate euphotic depth when decreasing PAR according to Kd when it reaches the 1% value
    euphotic_depth = 4.6052/kd_data
    # Now to compute the photolimit.
    threshold = 0.0035  # in Einstein m^-2 d^-1 (from 3.5mmol photons)
    threshold_depth = ( (np.log(par_data) - np.log(threshold)) / kd_data)

    # Plot the two depth dataset on a world map
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot euphotic depth
    ax1 = axes[0]
    ax1.set_global()
    ax1.coastlines(resolution='110m', linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    sc1 = ax1.pcolormesh(
        lon_grid, lat_grid, euphotic_depth, cmap='plasma', transform=ccrs.PlateCarree(),vmin=0, vmax=300)
    cbar = plt.colorbar(sc1, ax=ax1, orientation='vertical', label='Euphotic Depth (m)')
    cbar.ax.tick_params(labelsize=15)  # Set font size for the colorbar ticks
    ax1.set_title('Zeu i.e. 1% of PAR(0)', fontsize=16)

    # Plot threshold depth
    ax2 = axes[1]
    ax2.set_global()
    ax2.coastlines(resolution='110m', linewidth=1)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    sc2 = ax2.pcolormesh(
        lon_grid, lat_grid, threshold_depth, cmap='plasma', transform=ccrs.PlateCarree(),vmin=0, vmax=300)
    cbar = plt.colorbar(sc2, ax=ax2, orientation='vertical', label='Threshold Depth (m)')
    cbar.ax.tick_params(labelsize=15)  # Set font size for the colorbar ticks
    ax2.set_title('3.5 mmol photons', fontsize=17)
    plt.suptitle(f'Comparison for {date_str}', fontsize=20)

    plt.tight_layout()
    plt.savefig(f'euphotic_depth_comparison_{date_str}.png', dpi=300)
    plt.show()


    # Make a figure of the difference between the two depths
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    diff_depth = euphotic_depth - threshold_depth
    sc_diff = ax.pcolormesh(
        lon_grid, lat_grid, diff_depth, cmap='coolwarm', transform=ccrs.PlateCarree(), vmin=-130, vmax=10)
    plt.colorbar(sc_diff, ax=ax, orientation='vertical', label='Difference in Depth (m)')
    ax.set_title('Difference between Zeu and Photosynthesis threshold Depth', fontsize=16)
    plt.suptitle(f'Difference for {date_str}', fontsize=20)
    plt.tight_layout()
    plt.show()
