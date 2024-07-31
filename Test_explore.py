import glob
import os
import re
import sys

import random
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import seabass_maker as sb
sys.path.append('/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve')
import Toolbox_RAMSES as tools
import matplotlib.pyplot as plt
import Function_KD
import gsw
import Organelli_QC
import matplotlib.gridspec as gridspec
import subprocess
root = '/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve'
Processed_profiles = '/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve/Outputs'


# %% Download all the profiles from the floats from the GDAC

# First, have to download all the profiles from the floats from the GDAC, will be done using terminal commamd.
# Also will need to update floats that have new profiles since last processing.

df = pd.read_table(os.path.join(root, 'WMOvsNSerie.txt'))
list_wmo = df['WMO'].unique() # Floats we know to be hyperspectral that we process in this code

# Look at which floats we have as directories, if new floats were added, create according directories.
for number in list_wmo:
    # Create directory path
    dir_path = os.path.join(root, str(number))

    # Check if directory exists, if not create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, 'profiles_general'))
        print(f"Created profile directory: {dir_path}")

    command = f"wget -r -np --wait 1 -nH  -N --cut-dirs=3 -P {dir_path}  --reject 'index.html*' https://data-argo.ifremer.fr/aux/coriolis/{number}/"
    command_general = f"wget -r -np --wait 1 -nH  -N --cut-dirs=4 -P {os.path.join(dir_path, 'profiles_general')}  --reject 'index.html*' --accept 'R*.nc' https://data-argo.ifremer.fr/dac/coriolis/{number}/profiles/"
    print(command)

    subprocess.run(command_general, shell=True)

#Extract complete list of hyperspectral floats from a file

# http://www.argodatamgt.org/Access-to-data/Argo-GDAC-ftp-https-and-s3-servers

# %% Get WMO - automatically go through the list of all floats
wmos = []

for item in os.listdir(root):
    full_path = os.path.join(root, item)
    # Check if the item is a directory and matches the regular expression
    if os.path.isdir(full_path) and re.match("^\d+$", item):
        wmos.append(item)


# %% Read Calibration Files for all WMOs

df = pd.read_table(os.path.join(root, 'WMOvsNSerie.txt'))
cals = pd.DataFrame()

for wmo in wmos:
    for rad in ['Ed', 'Lu']:
        sn = df.N_Serie[(df.WMO == int(wmo)) & (df.EdLu == rad)].iloc[0]
        if pd.isna(sn):
            print(f'No Serial number for {rad}' + ' for float ' + wmo + ' in the database.')
        try:
            calibration_file = glob.glob(os.path.join(root, 'cals', f'*_{sn}_AllCal.txt'))[0]
            new_row = pd.DataFrame({'rad': [rad], 'calibration_file': [calibration_file], 'wmo': [wmo]})
            cals = pd.concat([cals, new_row], ignore_index=True)
        except IndexError:
            print(f'No calibration file for {rad}' + ' for float ' + wmo + ' in the database.')

# %% Read and Process Profiles

def zmld_boyer(s, t, p):
    """
    Computes mixed layer depth, based on de Boyer Montégut et al., 2004.

    Parameters
    ----------
    s : array_like
        salinity [psu (PSS-78)]
    t : array_like
        temperature [℃ (ITS-90)]
    p : array_like
        pressure [db].

    Notes
    -----
    Based on density with fixed threshold criteria
    de Boyer Montégut et al., 2004. Mixed layer depth over the global ocean:
        An examination of profile data and a profile-based climatology.
        doi:10.1029/2004JC002378

    dataset for test and more explanation can be found at:
    http://www.ifremer.fr/cerweb/deboyer/mld/Surface_Mixed_Layer_Depth.php

    Codes based on : http://mixedlayer.ucsd.edu/

    """
    # Remove NaN values
    valid_indices = ~np.isnan(s) & ~np.isnan(t) & ~np.isnan(p)
    s, t, p = s[valid_indices], t[valid_indices], p[valid_indices]

    s, t, p = s.reset_index(drop=True), t.reset_index(drop=True), p.reset_index(drop=True)


    m = len(s)

    if m <= 1:
        mldepthdens_mldindex = 0
        mldepthptemp_mldindex = 0
        return mldepthdens_mldindex, mldepthptemp_mldindex
    else:
        # starti = min(find((pres-10).^2==min((pres-10).^2)));
        starti = np.nanargmin((p - 10.0) ** 2)
        starti = 0
        pres = p[starti:m]
        sal = s[starti:m]
        temp = t[starti:m]

        pden = gsw.rho(sal, temp, pres) - 1000

        mldepthdens_mldindex = m - 1
        for i, pp in enumerate(pden):
            if np.abs(pden[starti] - pp) > 0.03:
                mldepthdens_mldindex = i
                break

        # Interpolate to exactly match the potential density threshold.
        presseg = [pres[mldepthdens_mldindex - 1], pres[mldepthdens_mldindex]]
        pdenseg = [
            pden[starti] - pden[mldepthdens_mldindex - 1],
            pden[starti] - pden[mldepthdens_mldindex],
        ]
        P = np.polyfit(presseg, pdenseg, 1)
        presinterp = np.linspace(presseg[0], presseg[1], 3)
        pdenthreshold = np.polyval(P, presinterp)

        # The potential density threshold MLD value:
        ix = np.max(np.where(np.abs(pdenthreshold) < 0.03)[0])
        mldepthdens_mldindex = presinterp[ix]

        # Search for the first level that exceeds the temperature threshold.
        mldepthptmp_mldindex = m - 1
        for i, tt in enumerate(temp):
            if np.abs(temp[starti] - tt) > 0.2:
                mldepthptmp_mldindex = i
                break

        # Interpolate to exactly match the temperature threshold.
        presseg = [pres[mldepthptmp_mldindex - 1], pres[mldepthptmp_mldindex]]
        tempseg = [
            temp[starti] - temp[mldepthptmp_mldindex - 1],
            temp[starti] - temp[mldepthptmp_mldindex],
        ]
        P = np.polyfit(presseg, tempseg, 1)
        presinterp = np.linspace(presseg[0], presseg[1], 3)
        tempthreshold = np.polyval(P, presinterp)

        # The temperature threshold MLD value:
        ix = np.max(np.where(np.abs(tempthreshold) < 0.2)[0])
        mldepthptemp_mldindex = presinterp[ix]

        return mldepthdens_mldindex, mldepthptemp_mldindex
def plot_ed_profiles(df, wmo, kd_df, wv_target, wv_og, ed0, depth_col='depth'):
    # Define the path to the figure

    # Extract the ED columns from the DataFrame
    ed_columns = [col for col in df.columns if col.startswith('ed')]
    kd_columns = [col for col in kd_df.columns if col.startswith('kd') and 'unc' not in col]
    kd_unc_columns = [col for col in kd_df.columns if col.startswith('kd') and 'unc' in col]
    ed0_columns = [col for col in ed0.columns if col.startswith('ed0')and 'unc' not in col]
    ed0_unc_columns = [col for col in ed0.columns if col.startswith('ed') and 'unc' in col]

    ed_wavelengths = np.array(wv_og)

    # Find the closest columns for each specified wavelength
    closest_columns = []
    closest_kd_columns = []
    closest_0_columns = []
    closest_indexs = []
    for wavelength in wv_target:
        if np.isnan(ed_wavelengths).all():
            closest_columns.append(np.nan)
        else:
            closest_index = np.nanargmin(np.abs(ed_wavelengths - wavelength))
            closest_indexs.append(closest_index)
            closest_column = ed_columns[closest_index]
            closest_columns.append(closest_column)
            closest_column = kd_columns[closest_index]
            closest_kd_columns.append(closest_column)
            closest_column = ed0_columns[closest_index]
            closest_0_columns.append(closest_column)

            # Iterate over each cycle
    for cycle in kd_df['profile'].unique():
        # Define the path to the figure
        figure_path = os.path.join(Processed_profiles,wmo,  f"{wmo}_{df[df['profile'] == cycle]['profile'].iloc[0]}_fig.png")

        # Check if the figure already exists
        if not os.path.exists(figure_path):
            # Check if all values for this cycle are NaN
            if kd_df[kd_df['profile'] == cycle][kd_columns].isna().all().all():
                print(f"All values for cycle {cycle} are NaN. Skipping figure creation.")
                continue

            # If the figure does not exist, generate it
            fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis='both', which='both', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0)

           # Check if the cycle is good or bad
            flag = kd_df[kd_df['profile'] == cycle]['quality_kd_489.0'].values[0]
            if flag == 0:
                status = 'PASSED'
            else:
                status = 'FAILED'

            fig.suptitle(f'Float {wmo} Cycle {cycle}: QC {status}', fontsize =20)  # Add the general title here
            gs = gridspec.GridSpec(3, 3)

            ax1 = fig.add_subplot(gs[0, :])  # First row, spans all columns
            ax2 = fig.add_subplot(gs[1, :])  # Second row, spans all columns
            ax3 = fig.add_subplot(gs[2, 0])  # Third row, first column
            ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column
            ax5 = fig.add_subplot(gs[2, 2])

            # Subplot for Ed(0-) values
            for inner_cycle in kd_df['profile'].unique():
                if inner_cycle != cycle and kd_df[kd_df['profile'] == inner_cycle]['quality_kd_489.0'].values[0] == 0 :
                    ax1.plot(wv_og, ed0[ed0['profile'] == inner_cycle][ed0_columns].values[0], color='lightgrey')
                    ax2.plot(wv_og, kd_df[kd_df['profile'] == inner_cycle][kd_columns].values[0], color='lightgrey')

            kd_values = kd_df[kd_df['profile'] == cycle][kd_columns].values[0]
            kd_unc_values = kd_df[kd_df['profile'] == cycle][kd_unc_columns].values[0]
            kd_unc_values = np.append(kd_unc_values, 0.1)
            kd_upper = kd_values + kd_unc_values
            kd_lower = kd_values - kd_unc_values
            ed0_unc_values = ed0[ed0['profile'] == cycle][ed0_unc_columns].values[0]
            ed0_unc_values = np.append(ed0_unc_values, 0.1)
            ed0_upper = ed0[ed0['profile'] == cycle][ed0_columns].values[0] + ed0_unc_values
            ed0_lower = ed0[ed0['profile'] == cycle][ed0_columns].values[0] - ed0_unc_values

            ax1.plot(wv_og, ed0[ed0['profile'] == cycle][ed0_columns].values[0], color='blue', linewidth=2)
            ax1.fill_between(wv_og, ed0_lower, ed0_upper, color='blue', alpha=0.2)
            ax2.plot(wv_og, kd_values, color='blue', linewidth=2)
            ax2.fill_between(wv_og, kd_lower, kd_upper, color='blue', alpha=0.2)

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Ed(0-) Values')
            ax1.set_title('Hyperspectral Ed(0-)')
            ax1.set_ylim([0.90 * min(ed0[ed0['profile'] == cycle][ed0_columns].values[0]),
                          1.20 * max(ed0[ed0['profile'] == cycle][ed0_columns].values[0])])
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Kd Values')
            ax2.set_title('Hyperspectral Kd')
            ax2.set_ylim([0.90 * min(kd_df[kd_df['profile'] == cycle][kd_columns].values[0]),
                          1.05 * max(kd_df[kd_df['profile'] == cycle][kd_columns].values[0])])

            # Plot the ED profiles
            colors = ['blue', 'green', 'red']  # Define a list of colors

            for idx, (ed_col, kd_col, ed0_col) in enumerate(
                    zip(closest_columns, closest_kd_columns, closest_0_columns)):
            # Use the specific flag to filter the df DataFrame
                if idx == 0:
                    ax = ax3
                elif idx == 1:
                    ax = ax4
                elif idx == 2:
                    ax = ax5

                kd_value = kd_df[kd_df['profile'] == cycle][kd_col].values[0]
                kd_unc_value = kd_df[kd_df['profile'] == cycle][kd_col + '_unc'].values[0]
                ed0_s = ed0[ed0['profile'] == cycle][ed0_col].values[0]
                ed0_unc_value = ed0[ed0['profile'] == cycle][ed0_col + '_unc'].values[0]

                df_filtered_flags = df[(df['profile'] == cycle) & (df['quality_kd_489.0'] == 0)]
                ax.scatter(df[df['profile'] == cycle][ed_col], df[df['profile'] == cycle][depth_col],
                           label=f'{ed_col}', c=colors[idx], alpha= 0.3 ,marker =  'x')
                ax.scatter(df_filtered_flags[ed_col],df_filtered_flags[depth_col],
                           label=f'{ed_col}', c = colors[idx])

                new_depth_values = np.linspace(df[df['profile'] == cycle][depth_col].min(), 50, len(df[depth_col]))
                ed_predicted = ed0_s * np.exp(-kd_value * new_depth_values)
                ed_predicted_upper = (ed0_s - ed0_unc_value) * np.exp(-(kd_value + kd_unc_value) * new_depth_values)
                ed_predicted_lower = (ed0_s + ed0_unc_value) * np.exp(-(kd_value - kd_unc_value) * new_depth_values)
                ax.plot(ed_predicted, new_depth_values, '--', label=f'Predicted ED from {kd_col}', color=colors[idx])
                ax.fill_betweenx(new_depth_values, ed_predicted_lower, ed_predicted_upper, color=colors[idx], alpha=0.2)
                ax.axhline(1 / kd_value, linestyle=':', color=colors[idx])
                ax.set_xlabel('ED Values')
                ax.set_ylabel('Depth (m)')
                ax.set_ylim([0, 50])
                ax.set_title(f'{ed_col} nm')
                ax.invert_yaxis()

            plt.tight_layout()
            plt.show()
            fig.savefig(figure_path)
        else:
            print(f"Figure already exists at {figure_path}")

# Bootstrapping function
def bootstrap_fit_klu(df, n_iterations=100, fit_method='iterative'):
    bootstrap_results = []
    bootstrap_Ed0 = []

    for i in range(n_iterations):
        # Resample the DataFrame with replacement
        df_resampled = df.sample(frac=0.8, replace=False, random_state=i)
        df_resampled = df_resampled.reset_index(drop=True)

        # Run the fit_klu function
        result = Function_KD.fit_klu(df_resampled, fit_method='iterative', wl_interp_method='None', smooth_method='None',
                                     only_continuous_obs=False)
        bootstrap_results.append(result['Kl'].values)
        bootstrap_Ed0.append(result['Luf'].values)


    # Convert the list of results into a DataFrame
    bootstrap_results = np.array(bootstrap_results)
    bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=result.index)

    median_ed0 = pd.DataFrame(bootstrap_Ed0, columns=result.index).median()
    std_ed0 = pd.DataFrame(bootstrap_Ed0, columns=result.index).std()


    # Calculate the median and standard deviation across the bootstrap samples
    median_luf_kd = bootstrap_results_df.median()
    std_luf_kd = bootstrap_results_df.std()

    return median_luf_kd, std_luf_kd, median_ed0, std_ed0

# Bootstrapping for dpeth dependancy on speed
def bootstrap_speed_depth(df,  Speed,  n_iterations=100):
    bootstrap_results = []
    random_numbers = np.random.normal(loc=0, scale=1, size=n_iterations)

    for i in random_numbers:
        # Resample the DataFrame with replacement
        df_resampled = df.copy()
        df_resampled['depth'] = df_resampled['depth'] + Speed * i
        # Run the fit_klu function
        result = Function_KD.fit_klu(df_resampled, fit_method='iterative', wl_interp_method='None', smooth_method='None',
                                     only_continuous_obs=False)
        bootstrap_results.append(result['Kl'].values)

    # Convert the list of results into a DataFrame
    bootstrap_results = np.array(bootstrap_results)
    bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=result.index)

    # Calculate the median and standard deviation across the bootstrap samples
    median_luf_depth = bootstrap_results_df.median()
    std_luf_depth = bootstrap_results_df.std()

    return median_luf_depth, std_luf_depth

data_Kd = []
data_Ed0 = []

for wmo in cals[(cals['rad'] == 'Ed')]['wmo']:

    # Check if the float has profiles
    if not os.path.exists(os.path.join(root, wmo, 'profiles')):
        print(f'No profiles for float {wmo}')
        continue

    # Check if the processed folder exists for a given float
    if not os.path.exists(os.path.join(Processed_profiles, wmo)):
        print(f'Processed folder did not exist for float {wmo}. Creating now...')
        os.makedirs(os.path.join(Processed_profiles, wmo))
    else:
        print(f'Processed folder exists for float {wmo}. Retrieving Ed and Kd files')
        try:
            Ed_physic = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed.csv')))
            Lu_physic = pd.DataFrame()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            Ed_physic = pd.DataFrame()
            print(f'Retrieval failed for float {wmo} Ed. Creating new dataframe...')

        try:
            Kd = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Kd.csv')))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            columns_data = {'profile': [], 'date': [], 'lon': [], 'lat': []}
            # Use dictionary comprehension to add columns '0' to '139' with pd.NA as their values
            columns_data.update({str(i): pd.NA for i in range(140)})
            # Create the DataFrame
            Kd = pd.DataFrame(columns_data)
            print(f"Retrieval failed for float {wmo} Kd. Creating new dataframe...")

        try:
            Ed0 = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed0.csv')))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            columns_data = {'profile': [], 'date': [], 'lon': [], 'lat': []}
            # Use dictionary comprehension to add columns '0' to '139' with pd.NA as their values
            columns_data.update({str(i): pd.NA for i in range(140)})
            # Create the DataFrame
            Ed0 = pd.DataFrame(columns_data)
            print(f"Retrieval failed for float {wmo} Ed0. Creating new dataframe...")

    # Check processed files
    processed_cycles = []
    processed_profiles = os.listdir(os.path.join(Processed_profiles, wmo))
    for file_name in processed_profiles:
        match = re.search(r"_([0-9]{3})_", file_name)
        if match:
            # Add the three middle numbers to the list
            processed_cycles.append(match.group(1))

    meta_filename = os.path.join(root, wmo, f'{wmo}_meta_aux.nc')

    # Set the maximum number of iterations
    # max_iterations = 15
    # # Loop counter
    # iteration_count = 0

    for idx, filename in enumerate(sorted(glob.glob(os.path.join(root, wmo, 'profiles', '*_aux.nc')))):

        current_cycle = re.search(r"_([0-9]+).*_aux\.nc$", filename).group(1)
        if current_cycle in processed_cycles and int(current_cycle) in Kd['profile'].values and int(current_cycle) in Ed_physic['profile']:
            print(f'Profile {current_cycle} already processed for float {wmo}. Skipping...')
            continue

        if '001D' in filename:
            print('Dark file, skipping') # Skip the file if it is a dark file
            continue

        # if (Kd['profile'] == int(current_cycle)).any() and (Ed_physic['profile'] == int(current_cycle)).any() and (
        #        Ed0['profile'] == int(current_cycle)).any():
        #     print(f'Profile {current_cycle} already processed for float {wmo}. Skipping...')
        #     continue

        # if iteration_count >= max_iterations:
        #     break  # Exit the loop if maximum iterations reached
        # iteration_count += 1

        data = xr.open_dataset(filename)

        # Grab the profile full name
        str_name = re.split('_aux.nc', os.path.basename(filename))
        base_data_name = os.path.join(root, wmo, 'profiles_general', str_name[0] + '.nc')
        try:
            data_base = xr.open_dataset(base_data_name)
        except FileNotFoundError:
            print('No general file')
            continue

        title = ('Float ' + wmo + ' Cycle ' + current_cycle)

        ed_n_prof = np.argwhere(
            data.STATION_PARAMETERS.values == b'RAW_DOWNWELLING_IRRADIANCE                                      ')
        lu_n_prof = np.argwhere(
            data.STATION_PARAMETERS.values == b'RAW_UPWELLING_RADIANCE                                          ')

        if not len(ed_n_prof) > 0 and len(lu_n_prof) > 0:
            print('skip')
            continue

        if np.isnan(data.RAW_DOWNWELLING_IRRADIANCE.values).all() and np.isnan(data.RAW_UPWELLING_RADIANCE.values).all():
            print('skip')
            continue

        skip_lu = True

        if len(lu_n_prof) == 0 or skip_lu == True:
            Ed_n_prof = ed_n_prof[0][0]
            try:
                 Ed_physic_profile = tools.format_ramses_ed_only( filename, meta_filename, cals[(cals['rad'] == 'Ed') &
                                                                                   (cals['wmo'] == wmo)]['calibration_file'].iloc[0], Ed_n_prof)
            except ValueError:
                print('Could not format Ed profile, stopping now')
                continue
        else:
            Ed_n_prof, Lu_n_prof = ed_n_prof[0][0], lu_n_prof[0][0]
            Ed_physic_profile, Lu_physic_profile = tools.format_ramses(
                filename, meta_filename, cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)]['calibration_file'].iloc[0],
                cals[(cals['rad'] == 'Lu') & (cals['wmo'] == wmo)]['calibration_file'].iloc[0], Ed_n_prof, Lu_n_prof)
            Ed_physic_profile = Ed_physic_profile.round(4)
            Lu_physic_profile = Lu_physic_profile.round(4)

        #Correct tilt from 10th of degree to 1 degree
        Ed_physic_profile['tilt'] = Ed_physic_profile['tilt'] / 10
        Ed_physic_profile['tilt_1id'] = Ed_physic_profile['tilt_1id'] / 10

        # Read Meta Data
        basename = os.path.basename(filename)
        metadata_ed = pd.DataFrame(  {'date': [data.JULD.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0],
                                    'wt': [np.nan] * Ed_physic_profile.shape[0],
                                    'sal': [np.nan] * Ed_physic_profile.shape[0],
                                    'lon': [data.LONGITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0],
                                    'lat': [data.LATITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0]})

        pres_ed = data.PRES.sel(N_PROF=Ed_n_prof).values[0:Ed_physic_profile.shape[0]]

        # Correct to the right timestamp
        metadata_ed.date = metadata_ed.date - data.MTIME.sel(N_PROF=Ed_n_prof).values[0: Ed_physic_profile.shape[0]]

        # Calculate Speed and PAR with depth
        Speed = np.full(len(metadata_ed), np.nan)
        for i in range(1, len(metadata_ed)):
            dt = (metadata_ed.date.iloc[i] - metadata_ed.date.iloc[i - 1]).total_seconds()
            dpres = pres_ed[i] - pres_ed[i - 1]
            if dt > 0:
                Speed[i] = dpres/dt # in m/s

        # # Compute difference of 2 seconds for the depth
        delta_depth = Speed * 2
        metadata_ed['depth'] = pres_ed - delta_depth
        metadata_ed.loc[0, 'depth'] = pres_ed[0] - delta_depth[1]

        # Interpolate the Base data so it is at the same depth as the optics data : No extrapolation is done
        interp_temp = np.interp(metadata_ed.depth, data_base.PRES.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]],
                                data_base.TEMP.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        interp_psal = np.interp(metadata_ed.depth, data_base.PRES.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]],
                                data_base.PSAL.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        metadata_ed['wt'] = interp_temp
        metadata_ed['sal'] = interp_psal

       # metadata_ed['MLD'] = zmld_boyer(metadata_ed.sal, metadata_ed.wt, metadata_ed.depth)[1]

        # Concatenate
        Ed_profile = pd.concat([metadata_ed, Ed_physic_profile], axis=1)

        # Extract wavelength to format for Kd function and rename columns
        wavelengths = [col for col in Ed_profile.columns if isinstance(col, (int, float))]
        # Round to 4 significant digits
        Ed_profile[wavelengths] = Ed_profile[wavelengths].round(4)

        # CALCUlATEE PAR
        # Convert irradiance to photon flux (micromol photons m⁻² s⁻¹ )
        irr_conv = (Ed_profile[wavelengths] * 1e-2) # Convert uW/cm/s-1 to W/m-2/nm
        const = np.array(wavelengths) * 1e-9 / 2.998 * 1e8 / 6.62606957e-34  #Convert wv to m and divide by speed of light/plank constant

        # Integrate photon flux over the wavelength range to get PAR (micromol photons m⁻² s⁻¹)
        Ed_profile['Epar'] = np.trapz(irr_conv * const / 6.02214129e23)

        # Organelli QC
        results = Organelli_QC.organelli16_qc(Ed_profile, lat=Ed_profile.lat[0],
                                                          lon=Ed_profile.lon[0],qc_wls=wavelengths , step2_r2=0.995, step3_r2=0.998,
                                                          skip_meta_tests=False, skip_dark_test=True)

        # Create an empty DataFrame with wavelengths as columns
        df_flags = pd.DataFrame(columns= wavelengths)
        df_results = pd.DataFrame({
            'global_flag': [np.nan],
            'status': [np.nan],
            'polynomial_fit': [np.nan],
            'wavelength': [np.nan]
        })
        # Iterate over the results
        for result in results:
            # Extract the wavelength and flags
            global_flag, flags, status, polynomial_fit, wv = result

            # Add the flags to the corresponding column in the DataFrame
            df_flags[wv] = flags
            # Add these values as a new row to the DataFrame
            new_row = pd.DataFrame({
                'global_flag': [global_flag],
                'status': [status],
                'polynomial_fit': [polynomial_fit],
                'wavelength': [wv]
                })
            df_results = pd.concat([df_results, new_row], ignore_index=True)

            df_results = df_results.dropna(how='all').reset_index(drop=True)

        # Find the wavelength closest to 489.0
        closest_wavelength = df_results['wavelength'].iloc[(df_results['wavelength'] - 489.0).abs().argmin()]

        if closest_wavelength in df_results['wavelength'].values:
            status_value = df_results.loc[df_results['wavelength'] == closest_wavelength, 'status'].values[0]
            if df_results.loc[df_results['wavelength'] == closest_wavelength, 'global_flag'].values[0] == False:
                print(
                    f"Cycle {current_cycle} failed QC at {closest_wavelength} for {df_results.loc[df_results['wavelength'] == closest_wavelength, 'status'].values[0]}: Careful proceeding")
                Ed_profile[f'quality_kd_{closest_wavelength}'] = 2  # Bad flags at closest wavelength, don't use!
            else:
                Ed_profile[f'quality_kd_{closest_wavelength}'] = df_flags[
                    closest_wavelength]  # Good flags at closest wavelength, use with confidence!
        else:
            print(f"Wavelength {closest_wavelength} not QC. Bad profile.")
            Ed_profile[f'quality_kd_{closest_wavelength}'] = 2  # Bad flags at closest wavelength, don't use!
            status_value = 'BAD'  # Set a default value or skip the operation

        for col in wavelengths:
            Ed_profile.rename(columns={col: 'ed' + str(col)}, inplace=True)

        # Select rows to use
        idx_end = len(Ed_profile) - 1
        lu_columns = [col for col in Ed_profile.columns if col.startswith(('ed', 'lu'))]

        ###### Create Kd document ######
       # flag_idx = df_flags.isin([2])  # Omit rows with flags 2 (BAD)

        new_Ed = Ed_profile.loc[:,['date', 'depth'] + lu_columns].copy()
        new_Ed = new_Ed.rename(columns={'date': 'datetime' })

        # Iterate over each wavelength column in new_Ed
        for wavelength, column in zip(wavelengths, new_Ed.columns):
            # Check if the wavelength is below 600
            if wavelength < 650:
                # Get the flag values for this wavelength
                flags = df_flags[wavelength]
                # Replace the values in new_Ed where the flag is 2 with np.nan
                new_Ed.loc[flags == 2, column] = np.nan

        # Add to global table of the float
        new_column_names = ["kd" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Kd.columns[4:len(wavelengths)+4], new_column_names))
        Kd = Kd.rename(columns=column_mapping)

        # Generate new column names with "_unc" for the following 70 columns
        new_column_names_unc = ["kd" + str(wavelength) + "_unc" for wavelength in wavelengths]
        # Map these new names with "_unc" to the columns 74 to 143
        column_mapping_unc = dict(zip(Kd.columns[len(wavelengths)+5 :len(wavelengths)*2 +5], new_column_names_unc))
        # Apply the renaming for the second set of columns
        Kd = Kd.rename(columns=column_mapping_unc)

        new_column_names = ["ed0" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Ed0.columns[4:75], new_column_names))
        Ed0 = Ed0.rename(columns=column_mapping)
        new_column_names_unc = ["ed0" + str(wavelength) + "_unc" for wavelength in wavelengths]
        # Map these new names with "_unc" to the columns 74 to 143
        column_mapping_unc = dict(zip(Ed0.columns[len(wavelengths)+5 :len(wavelengths)*2 +5], new_column_names_unc))
        # Apply the renaming for the second set of columns
        Ed0 = Ed0.rename(columns=column_mapping_unc)

        # kd_res = Function_KD.fit_klu(new_Ed, fit_method='iterative', wl_interp_method='None', smooth_method='None',
        #                              only_continuous_obs=False)

        # Calculate the median and standard deviation across the bootstrap samples
        median_luf_kd, std_luf_kd, median_Ed0, std_Ed0 = bootstrap_fit_klu(new_Ed, n_iterations=100, fit_method='iterative')
        median_luf_depth, std_luf_depth = bootstrap_speed_depth(new_Ed, Speed, n_iterations=100)

        # Calculate uncertainties
        Kd_uncertainty = (std_luf_kd**2 + std_luf_depth**2)**0.5


        # Set 'TILT_FLAG' in data_dict based on the calculated percentage
        # tilt_flag_value = 1 if tilt_flag_percentage > 0.5 else 0

        data_dict_K ={
            'profile': int(current_cycle),
            'date': Ed_profile.date[0],
            'lon': Ed_profile.lon[0].round(5),
            'lat': Ed_profile.lat[0].round(5),
            'quality_kd_489.0': status_value
        }

        data_dict_K.update(dict(zip(Kd.columns[4: len(wavelengths) +4],median_luf_kd)))
        data_dict_K.update(dict(zip(Kd.columns[len(wavelengths) +5: len(wavelengths)*2 +5], Kd_uncertainty.values.reshape(-1).astype(np.float32))))

        data_Kd.append(data_dict_K)

        data_dict ={
             'profile': int(current_cycle),
            'date': Ed_profile.date[0],
            'lon': Ed_profile.lon[0].round(5),
            'lat': Ed_profile.lat[0].round(5),
            'quality_kd_489.0': status_value
        }

        data_dict.update(dict(zip(Ed0.columns[4: len(wavelengths) +4], median_Ed0)))
        data_dict.update(dict(zip(Ed0.columns[len(wavelengths) + 5: len(wavelengths) * 2 + 5],
                                    std_Ed0.values.reshape(-1).astype(np.float32))))

        data_Ed0.append(data_dict)

        # save csv file
        Ed_profile.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_' + current_cycle + '_Ed.csv'), index=False)
        print(f" Ed profile for float {wmo} Cycle {current_cycle} was created")

        Ed_with_station = Ed_profile.copy()
        Ed_with_station['profile'] = current_cycle
        Ed_physic = pd.concat([Ed_physic, Ed_with_station])

    Kd = pd.DataFrame(data_Kd)
    Kd = Kd.round(5)

    Ed0 = pd.DataFrame(data_Ed0)
    Ed0 = Ed0.round(5)

    Ed_physic = Ed_physic.round(5)
    Ed_physic = Ed_physic.round(5)
    if not Kd.empty:
        # Replace 'GOOD' with 0
        Kd['quality_kd_489.0'] = Kd['quality_kd_489.0'].replace('GOOD', 0)
        # Replace values that start with 'BAD' with 2
        Kd.loc[Kd['quality_kd_489.0'] != 0, 'quality_kd_489.0'] = 2

        Kd.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Kd.csv'), index=False)
        print(f'Kd file for float {wmo} was created')
        Ed0.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed0.csv'), index=False)
        print(f'Ed0 file for float {wmo} was created')
        Ed_physic.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed.csv'), index=False)
        print(f'Ed file for float {wmo} was created')

# %% QUALITY CONTROL OF ALL KD PROFILES FROM A SINGLE FLOAT

# Load the Kd profiles
Kd = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Kd.csv')))
Ed0 = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed0.csv')))
Ed_all = pd.read_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed.csv'))

def extracted_wavelengths(Ed, pattern='ed'):
    # Extract wavelengths from the Kd DataFrame
    columns = [col for col in Ed.columns if col.startswith(pattern)]

    wavelengths = []
    for col in columns:
        match = re.search(pattern + r'(\d+\.?\d*)', col)
        if match:
            wavelengths.append(float(match.group(1)))
        else:
            wavelengths.append(np.nan)
    wavelengths = np.array(wavelengths)
    return wavelengths

wavelengths = extracted_wavelengths(Ed0, pattern='ed')

plot_ed_profiles(Ed_all,wmo, Kd, [490, 550, 660], wavelengths, Ed0, depth_col='depth')

kd_columns = [col for col in Kd.columns if col.startswith('KD')]
# Calculate the mean Kd and the mean Ed0 for each wavelength
mean_kd = Kd[kd_columns].mean()
mean_ed0 = Ed0[[col.replace('KD', 'Ed0') for col in kd_columns]].mean()

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['quality_kd_489.0'] ==0]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.CYCLE}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
plt.ylim([0, 1])
fig.suptitle('Good Kd profiles for float ' + wmo)
plt.show()

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['quality_kd_489.0'] ==2]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.CYCLE}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
plt.ylim([0, 3])
fig.suptitle('BAD Kd profiles for float ' + wmo)
plt.show()

# %%  Make the seabass file

Kd = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Kd.csv')))
Ed_41 = pd.read_csv(os.path.join(Processed_profiles, wmo, wmo + '_041_Ed.csv'))

# Define metadata for the SeaBASS file
metadata = {
    'investigators': 'Nils_Haentjens, Charlotte_Begouen_Demeaux, others?',
    'affiliations': 'University_of_Maine, University_of_Maine',
    'contact': 'nils.haentjens@maine.edu',
    'experiment': 'PVST-VDIUP',
    'cruise': wmo,
    'profile': 'NA',
    'documents': 'none',
    'calibration_files': 'cals/SAM_876E_01600042_AllCal.txt',
    'data_type': 'Profiling_float',
    'data_status': 'preliminary',
    'water_depth': 'NA',
    'measurement_depth': 'NA',
}

filename = os.path.join(Processed_profiles, wmo, (wmo + '_041_Ed'))
sb.format_to_seabass(Ed_41, metadata, filename, missing_value_placeholder='-9999', delimiter='comma')
