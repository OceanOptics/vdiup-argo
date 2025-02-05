import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import seabass_maker as sb
sys.path.append('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve')
import Toolbox_RAMSESv2 as tools
import matplotlib.pyplot as plt
import Function_KD
import gsw
import Organelli_QC_Shapiro
import matplotlib.gridspec as gridspec
import subprocess
import time
root = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve'
Processed_profiles = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs_5_QC'


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
    subprocess.run(command, shell=True)

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
def plot_ed_profiles(df, wmo, kd_df, wv_target, wv_og, ed0, flags_df, depth_col='depth'):
    # Define the path to the figure

    kd_df.replace('-9999', np.nan, inplace=True)
    ed0.replace('-9999', np.nan, inplace=True)
    df = df.dropna(subset=['depth']) # Don't want nan in depth col

    # Extract the ED columns from the DataFrame
    ed_columns = [col for col in df.columns if col.startswith('ed')]
    kd_columns = [col for col in kd_df.columns if col.startswith('kd') and 'unc' not in col and '_se' not in col and '_bincount' not in col]
    kd_unc_columns = [col for col in kd_df.columns if col.startswith('kd') and 'unc' in col]
    ed0_columns = [col for col in ed0.columns if col.startswith('ed0')and 'unc' not in col]
    ed0_unc_columns = [col for col in ed0.columns if col.startswith('ed') and 'unc' in col]
    quality_col = [col for col in kd_df.columns if col.startswith('quality')]
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
        if cycle in pd.to_numeric(df['profile']):
            profile_number = str(df[pd.to_numeric(df['profile']) == cycle]['profile'].iloc[0]).zfill(3)
        else:
            continue
        figure_path = os.path.join(Processed_profiles, wmo, f"{wmo}_{profile_number}_fig.png")

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
            flag = kd_df[kd_df['profile'] == cycle][quality_col].values[0]
            if flag == 0:
                status = 'PASSED'
            elif flag == 1:
                status = 'QUESTIONABLE'
            elif flag == 2:
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
                if inner_cycle != cycle and kd_df[kd_df['profile'] == inner_cycle][quality_col].values[0] == 0:
                    ax1.plot(wv_og, ed0[ed0['profile'] == inner_cycle][ed0_columns].values[0], color='lightgrey')
                    ax2.plot(wv_og, kd_df[kd_df['profile'] == inner_cycle][kd_columns].values[0], color='lightgrey')

            kd_values = pd.to_numeric(kd_df[kd_df['profile'] == cycle][kd_columns].values[0])
            kd_unc_values = pd.to_numeric(kd_df[kd_df['profile'] == cycle][kd_unc_columns].values[0], errors='coerce')
            kd_upper = kd_values + kd_unc_values
            kd_lower = kd_values - kd_unc_values
            ed0_unc_values = ed0[ed0['profile'] == cycle][ed0_unc_columns].values[0]
            ed0_upper = ed0[ed0['profile'] == cycle][ed0_columns].values[0] + ed0_unc_values
            ed0_lower = ed0[ed0['profile'] == cycle][ed0_columns].values[0] - ed0_unc_values
            ed0_values = ed0[ed0['profile'] == cycle][ed0_columns].values[0]


            ax1.plot(wv_og, ed0_values, color='blue', linewidth=2)
            ax1.fill_between(wv_og, ed0_lower, ed0_upper, color='blue', alpha=0.2)
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Ed(0-) Values')
            ax1.set_title('Hyperspectral Ed(0-)')
            m_val = ed0[ed0['profile'] == cycle][ed0_columns].values[0]
            ax1.set_ylim([0.90 * min(m_val[~np.isnan(m_val)]),
                          1.20 * max(m_val[~np.isnan(m_val)])])
            ax1.set_xlim([min(wv_og), 700])

            ax2.plot(wv_og, kd_values, color='blue', linewidth=2)
            ax2.fill_between(wv_og, kd_lower, kd_upper, color='blue', alpha=0.2)
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Kd Values')
            ax2.set_title('Hyperspectral Kd')
            ax2.set_ylim([0.90 * min(kd_values[~np.isnan(kd_values)]),
                          1.05 * max(kd_values[~np.isnan(kd_values)])])
            ax2.set_xlim([min(wv_og), 700])

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
                # Filter the DataFrame once

                flags_prof = flags_df[(pd.to_numeric(flags_df['profile']) == cycle)]
                df_filtered = df[pd.to_numeric(df['profile']) == cycle]

                flags = flags_prof[f'flag_{ed_wavelengths[closest_indexs[idx]]}'].reset_index(drop=True).reindex(df_filtered.index)
                good_flags = flags[flags == 0].index
                question_flags = flags[flags == 1].index

                # Scatter plot for all data points
                ax.scatter(df_filtered[ed_col], df_filtered[depth_col],
                           label=f'{ed_col}', c=colors[idx], alpha=0.3, marker='x')
                ax.scatter(df_filtered[ed_col][question_flags], df_filtered[depth_col][question_flags],
                       label=f'Flagged {ed_col}', c=colors[idx], alpha=0.3, marker='o')
                ax.scatter(df_filtered[ed_col][good_flags], df_filtered[depth_col][good_flags],
                           label=f'Flagged {ed_col}', c=colors[idx], alpha=0.7, marker='o')

                new_depth_values = np.linspace(df[pd.to_numeric(df['profile']) == cycle][depth_col].min(), 50, len(df[depth_col]))
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
def bootstrap_fit_klu_depth(df, Speed, n_iterations=10, fit_method='iterative'):
    bootstrap_results = []
    bootstrap_Ed0 = []
    random_numbers = np.random.normal(loc=0, scale=1, size=n_iterations)


    for i in range(n_iterations):
        # Resample the DataFrame with replacement
        df_resampled = df.copy()
        df_resampled['depth'] = df_resampled['depth'] + Speed * random_numbers[i]

        for col in df.columns:
            if col.startswith('ed'):
                # Identify non-NaN values
                non_nan_indices = df[col].dropna().index
                nan_indices = pd.Series(non_nan_indices).sample(frac=0.2, random_state=i)
                df_resampled.loc[nan_indices, col] = np.nan

        # Run the fit_klu function
        try:
            result = Function_KD.fit_klu(df_resampled, fit_method='iterative', wl_interp_method='None',
                                         smooth_method='None', only_continuous_obs=False)
            bootstrap_results.append(result['Kl'].values)
            bootstrap_Ed0.append(result['Luf'].values)
        except Exception as e:
            print(f"An error occurred during bootstrap iteration {i}: {e}")
            continue

    if np.isnan(bootstrap_results).all():
        median_luf_kd = [np.nan] * len(wavelengths)
        std_luf_kd = [np.nan] * len(wavelengths)
        median_ed0 = [np.nan] * len(wavelengths)
        std_ed0 = [np.nan] * len(wavelengths)
    else:
        # Convert the list of results into a DataFrame
        bootstrap_results = np.array(bootstrap_results)
        bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=result.index)

        median_ed0 = pd.DataFrame(bootstrap_Ed0, columns=result.index).median()
        std_ed0 = pd.DataFrame(bootstrap_Ed0, columns=result.index).std()

        # Calculate the median and standard deviation across the bootstrap samples
        median_luf_kd = bootstrap_results_df.median()
        std_luf_kd = bootstrap_results_df.std()

    return median_luf_kd, std_luf_kd, median_ed0, std_ed0

# Function to find the closest wavelengths
def find_closest_wavelengths(targets, available_wavelengths):
    closest_wavelengths = []
    for target in targets:
        closest = min(available_wavelengths, key=lambda x: abs(x - target))
        closest_wavelengths.append(closest)
    return closest_wavelengths

Check_5wv =[]
combined_QC_5wv = pd.DataFrame()

# Define the target wavelengths
target_qc_wavelengths = [380, 443, 490, 550, 620]

for wmo in cals[(cals['rad'] == 'Ed')]['wmo']:

    data_Kd, data_Ed0 = [], []
    data_flags = pd.DataFrame()

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
        print(f'Retrieval failed for float {wmo} Ed. Creating new dataframe...')
        Ed_physic = pd.DataFrame()
    try:
        Kd = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Kd.csv')))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        columns_data = {'profile': [], 'date': [], 'time': [], 'lon': [], 'lat': [], 'quality': []}
        # Use dictionary comprehension to add columns '0' to '139' with pd.NA as their values
        columns_data.update({str(i): pd.NA for i in range(282)})
        # Create the DataFrame
        Kd = pd.DataFrame(columns_data)
        print(f"Retrieval failed for float {wmo} Kd. Creating new dataframe...")

    try:
        Ed0 = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed0.csv')))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        columns_data = {'profile': [], 'date': [], 'time': [], 'lon': [], 'lat': []}
        # Use dictionary comprehension to add columns '0' to '139' with pd.NA as their values
        columns_data.update({str(i): pd.NA for i in range(141)})
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

    # Check if the calibration file exists for the given wmo and Ed
    if  cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)].empty:
        print(f"No calibration file found for rad='Ed' and wmo={wmo}")
        continue


    comments = ['These data were collected and made freely available by the International Argo Program and the national programs',
        'that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the',
        'Global Ocean Observing System https://doi.org/10.17882/42182.',
        f'Link to BGC-Argo GDAC for raw float data: https://data-argo.ifremer.fr/aux/coriolis/{wmo}.',
        'Quality Flag relates to the overall radiometric quality control based on Organelli et al., 2016 (DOI: 10.1175/JTECH-D-15-0193.1).',
        'Quality control is performed at each wavelength, see documentation for details.',
        'The overall "quality" flag per profile is recorded based on performance of all wavelengths below 600nm with following definition:',
        '    0. Good: >50% of wavelengths passed individual QC.',
        '    1. Questionable: >50% of wavelengths are questionable following individual QC or >5% of wavelengths flagged as Bad.',
        '    2. Bad: >50% of wavelengths are bad following individual QC.',
        'Uncertainties (_unc) are computed with a bootstrap technique and encompass uncertainty in fitting Kd to the profile and depth uncertainty. Details available in documentation.']

    # Define metadata for the SeaBASS file
    metadata = {'investigators': 'Nils_Haentjens,Charlotte_Begouen_Demeaux',
        'affiliations': 'University_of_Maine,University_of_Maine',
        'contact': 'nils.haentjens@maine.edu',
        'experiment': 'PVST_VDIUP',
        'cruise': 'VDIUP-Argo-Kd',
        'platform_id': wmo,
        'instrument_manufacturer': 'TriOS',
        'instrument_model': 'RAMSES',
        'documents': 'PVST_VDIUP_float_documentation.pdf',
        'calibration_files': os.path.basename(
            cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)]['calibration_file'].iloc[0]),
        'data_type': 'drifter',
        'data_status': 'preliminary',
        'water_depth': 'NA',
        'measurement_depth': 'NA'}

    # Initialize an empty DataFrame to store the results

    for idx, filename in enumerate(sorted(glob.glob(os.path.join(root, wmo, 'profiles', '*_aux.nc')))):

        current_cycle = re.search(r"_([0-9]+).*_aux\.nc$", filename).group(1)
        # if current_cycle in processed_cycles and int(current_cycle) in Kd['profile'].values and int(current_cycle) in Ed_physic['profile']:
        #      print(f'Profile {current_cycle} already processed for float {wmo}. Skipping...')
        #      continue

        if '001D' in filename:
            print('Dark file, skipping') # Skip the file if it is a dark file
            continue

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
        # Location has changed in the new version of the data
        if len(ed_n_prof) == 0:
            ed_n_prof = np.argwhere(
                data.PARAMETER.values == b'RAW_DOWNWELLING_IRRADIANCE                                      ')


        if not len(ed_n_prof) > 0:
            print('skip')
            continue

        if 'RAW_DOWNWELLING_IRRADIANCE' in data and 'RAW_UPWELLING_RADIANCE' in data:
            if np.isnan(data.RAW_DOWNWELLING_IRRADIANCE.values).all() and np.isnan(
                    data.RAW_UPWELLING_RADIANCE.values).all():
                print('skip')
                continue
        else :
            print('skip')
            continue

        skip_lu = True

        if  skip_lu == True:
            Ed_n_prof = ed_n_prof[0][0]
            try:
                if '1903578' in filename:
                    Ed_physic_profile = tools.format_ramses_ed_only(filename, meta_filename,
                                                                    cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)][
                                                                        'calibration_file'].iloc[0], Ed_n_prof,
                                                                    PixelStop=144)
                # else if float is 4903660 , and cycle is 14 or more
                elif '4903660' in filename and int(current_cycle) >= 13:
                    Ed_physic_profile = tools.format_ramses_ed_only(filename, meta_filename,
                                                                    cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)][
                                                                        'calibration_file'].iloc[0], Ed_n_prof,
                                                                    PixelBinning = 1)
                else:
                    Ed_physic_profile = tools.format_ramses_ed_only(filename, meta_filename,
                                                                    cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)][
                                                                        'calibration_file'].iloc[0], Ed_n_prof)
            except ValueError:
                print('Could not format Ed profile from counts, skipping')
                continue

        columns_to_check = [col for col in Ed_physic_profile.columns if col not in ['tilt', 'tilt_1id']]
        if Ed_physic_profile[columns_to_check].map(lambda x: pd.isna(x) or np.isinf(x)).all().all():
            print('All values in the relevant columns are NaN or infinite. Skipping...')
            continue

        #Correct tilt from 10th of degree to 1 degree
        Ed_physic_profile['tilt'] = Ed_physic_profile['tilt'] / 10
        Ed_physic_profile['tilt_1id'] = Ed_physic_profile['tilt_1id'] / 10
        # Read Meta Data
        basename = os.path.basename(filename)
        metadata_ed = pd.DataFrame(  {'wt': [np.nan] * Ed_physic_profile.shape[0],
                                    'sal': [np.nan] * Ed_physic_profile.shape[0],
                                    'lon': [data.LONGITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0],
                                    'lat': [data.LATITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0]})


        DT = [data.JULD.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0]
        pres_ed = data.PRES.sel(N_PROF=Ed_n_prof).values[0:Ed_physic_profile.shape[0]]

        # Correct to the right timestamp
        DT = DT - data.MTIME.sel(N_PROF=Ed_n_prof).values[0: Ed_physic_profile.shape[0]]
        # Separate the date and time
        DT = pd.to_datetime(DT)
        metadata_ed['date'] = DT.strftime('%Y%m%d')
        metadata_ed['time'] = DT.strftime('%H:%M:%S')

        # Calculate Speed and PAR with depth
        Speed = np.full(len(metadata_ed), np.nan)
        if len(metadata_ed) == 1:
            print('Ed profile only has 1 depth, skipping')
            continue
        for i in range(1, len(metadata_ed)):
            dt = (DT[i] - DT[i - 1]).total_seconds()
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

        # Concatenate
        Ed_profile = pd.concat([metadata_ed, Ed_physic_profile], axis=1)

        # Extract wavelength to format for Kd function and rename columns
        wavelengths = [col for col in Ed_profile.columns if isinstance(col, (int, float))]

        # CALCUlATE PAR
        # Convert irradiance to photon flux (micromol photons m⁻² s⁻¹ )
        irr_conv = (Ed_profile[wavelengths] * 10**-2) # Convert uW/cm/s-1 to W/m-2/nm
        const = np.array(wavelengths) * 1e-9 /(2.998e8 * 6.62606957e-34) #Convert wv to m and divide by speed of light * plank constant

        # Integrate photon flux over the wavelength range to get PAR (micromol photons m⁻² s⁻¹)
        I = (np.array(wavelengths) >= 350) & (np.array(wavelengths) <= 700)
        Ed_profile['Epar'] = np.trapz((np.array(irr_conv)[:,I] * const[I]) / 6.02214129e23 ) * 1e6 / 1e4 # in umol/cm-2/s-14

                # Organelli QC
        # Measure time for the first function
        start_time = time.time()
        results = Organelli_QC_Shapiro.organelli16_qc(Ed_profile, lat=Ed_profile.lat[0],
                                                      lon=Ed_profile.lon[0], qc_wls=wavelengths, step2_r2=0.995, step3_r2=0.997,
                                                      step3_r3=0.999, skip_meta_tests=False)
        end_time = time.time()
        time_full_wavelengths = end_time - start_time

        # Measure time for the second function

        qc_5wv = find_closest_wavelengths(target_qc_wavelengths, wavelengths)
        start_time = time.time()
        results_5wv = Organelli_QC_Shapiro.organelli16_qc(Ed_profile, lat=Ed_profile.lat[0],
                                                          lon=Ed_profile.lon[0], qc_wls=qc_5wv, step2_r2=0.995, step3_r2=0.997,
                                                          step3_r3=0.999, skip_meta_tests=False)
        end_time = time.time()
        time_5_wavelengths = end_time - start_time

        print(f"Time for full wavelengths: {time_full_wavelengths:.2f} seconds")
        print(f"Time for 5 wavelengths: {time_5_wavelengths:.2f} seconds")

# Process hyperspectral results first
        df_flags = pd.DataFrame(columns=wavelengths, index=range(len(Ed_profile)))
        df_results = pd.DataFrame({
            'global_flag': [np.nan],
            'status': [np.nan],
            'polynomial_fit': [np.nan],
            'wavelength': [np.nan]
        })

        for result in results:
            # Extract the wavelength and flags
            global_flag, flags, status, polynomial_fit, wv = result
            if len(flags) < len(df_flags):
                # Create a new array filled with NaN of the same length as df_flags
                new_flags = np.full(len(df_flags), 2)
                # Fill the top of this array with the flags data
                new_flags[:len(flags)] = flags
            else:
                new_flags = flags
                # Assign the new_flags array to the appropriate column in df_flags
            df_flags[wv] = new_flags
            # Add these values as a new row to the DataFrame
            new_row = pd.DataFrame({
                'global_flag': [global_flag],
                'status': [status],
                'polynomial_fit': [polynomial_fit],
                'wavelength': [wv]
                })
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results = df_results.dropna(how='all').reset_index(drop=True)

        data_dict_flags = {
            'depth': Ed_profile['depth'].values,
            'profile': [current_cycle] * len(Ed_profile)}

        for wavelength in wavelengths:
            data_dict_flags[f'flag_{wavelength}'] = df_flags[wavelength].values

        data_flags = pd.concat([data_flags, pd.DataFrame(data_dict_flags)], ignore_index=True)

        df_results_filtered = df_results[df_results['wavelength'] < 660]
        if ((df_results_filtered['global_flag'] == 2).sum() / len(df_results_filtered) >= 0.8 or
                (df_results_filtered['global_flag'] == 1).sum() / len(df_results_filtered) == 1):
            Ed_profile['quality'] = 2  #
            print(
                f"Cycle {current_cycle} fails QC for more than 80% of wavelength or all are questionnable: Careful proceeding : BAD")
        elif ((df_results_filtered['global_flag'] == 0).sum() / len(df_results_filtered) >= 0.8):
            Ed_profile['quality'] = 0  # GOOD
            print(f"Cycle {current_cycle} passes QC for more than 80% of wavelength: GOOD")
        else:
            Ed_profile['quality'] = 1
            print(f"Cycle {current_cycle} : Questionnable QC results")

        # Same for 5 wv_QC
        df_flags_5wv = pd.DataFrame(columns=qc_5wv, index=range(len(Ed_profile)))
        df_results_5wv = pd.DataFrame({
            'global_flag': [np.nan],
            'status': [np.nan],
            'polynomial_fit': [np.nan],
            'wavelength': [np.nan]
        })

        # Process results_5wv
        for result in results_5wv:
            # Extract the wavelength and flags
            global_flag, flags, status, polynomial_fit, wv = result
            if len(flags) < len(df_flags_5wv):
                # Create a new array filled with NaN of the same length as df_flags
                new_flags = np.full(len(df_flags_5wv), 2)
                # Fill the top of this array with the flags data
                new_flags[:len(flags)] = flags
            else:
                new_flags = flags
            # Assign the new_flags array to the appropriate column in df_flags
            df_flags_5wv[wv] = new_flags
            # Add these values as a new row to the DataFrame
            new_row = pd.DataFrame({
                'global_flag': [global_flag],
                'status': [status],
                'polynomial_fit': [polynomial_fit],
                'wavelength': [wv]
            })
            df_results_5wv = pd.concat([df_results_5wv, new_row], ignore_index=True)
            df_results_5wv = df_results_5wv.dropna(how='all').reset_index(drop=True)

        # Ensure all target wavelengths are present in df_results_5wv
        global_flags_5 = df_results_5wv.set_index('wavelength')['global_flag'].reindex(qc_5wv, fill_value=2).values
        # Create a DataFrame with the required columns
        temp_df = pd.DataFrame({
            'wmo': [wmo] ,
            'current_cycle': current_cycle,
            'wv_1': global_flags_5[0],
            'wv_2': global_flags_5[1],
            'wv_3': global_flags_5[2],
            'wv_4': global_flags_5[3],
            'wv_5': global_flags_5[4]
        })
        combined_QC_5wv = pd.concat([combined_QC_5wv, temp_df], ignore_index=True)

        # Count the occurrences of each global_flag
        flag_counts = df_results_5wv['global_flag'].value_counts()

        # Initialize the counts for each flag
        count_0 = flag_counts.get(0, 0)
        count_1 = flag_counts.get(1, 0)
        count_2 = flag_counts.get(2, 0)

        conditions = {
            (5, 0, 0): (0, "PASSED"),
            (4, 1, 0): (0, "PASSED"),
            (4, 0, 1): (1, "PASSED"),
            (3, 1, 1): (1, "QUESTIONABLE"),
            (3, 2, 0): (1, "QUESTIONABLE"),
            (3, 0, 2): (2, "BAD"),
            (2, 3, 0): (1, "QUESTIONABLE"),
            (2, 2, 1): (2, "QUESTIONABLE"),
            (2, 1, 2): (2, "BAD"),
            (2, 0, 3): (2, "BAD"),
            (1, 4, 0): (1, "QUESTIONABLE"),
            (1, 3, 1): (1, "QUESTIONABLE"),
            (1, 2, 2): (2, "BAD"),
            (1, 1, 3): (2, "BAD"),
            (1, 0, 4): (2, "BAD"),
            (0, 5, 0): (2, "BAD"),
            (0, 4, 1): (2, "BAD"),
            (0, 3, 2): (2, "BAD"),
            (0, 2, 3): (2, "BAD"),
            (0, 1, 4): (2, "BAD"),
            (0, 0, 5): (2, "BAD")
        }

        # Determine the quality based on the specified conditions
        quality_5wv, message = conditions.get((count_0, count_1, count_2), (1, "QUESTIONABLE"))
        Check_5wv.append({'wmo': wmo, 'cycle_number': current_cycle, 'quality_5wv': quality_5wv})

        for col in wavelengths:
            Ed_profile.rename(columns={col: 'ed' + str(col)}, inplace=True)

        # Select rows to use
        idx_end = len(Ed_profile) - 1
        lu_columns = [col for col in Ed_profile.columns if col.startswith(('ed', 'lu'))]

        ###### Create Kd document ######

        new_Ed = Ed_profile.loc[:,['date','time', 'depth'] + lu_columns].copy()

        # Iterate over each wavelength column in new_Ed
        for wavelength, column in zip(wavelengths, lu_columns):
            # Get the flag values for this wavelength
            flags = df_flags[wavelength]
            # Replace the values in new_Ed where the flag is 2 with np.nan
            new_Ed.loc[flags[flags == 2].index, column] = np.nan

        fileN = 'Ed_Argo_Hyperspectral_' + wmo + '_' + current_cycle
        path = os.path.join(Processed_profiles, wmo)

        # Add to global table of the float
        new_column_names = ["kd" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Kd.columns[6:len(wavelengths)+6], new_column_names))
        Kd = Kd.rename(columns=column_mapping)
        # Generate new column names with "_unc" for the following 70 columns
        new_column_names_unc = ["kd" + str(wavelength) + "_unc" for wavelength in wavelengths]
        # Map these new names with "_unc" to the columns 74 to 143
        column_mapping_unc = dict(zip(Kd.columns[len(wavelengths)+ 6: len(wavelengths)*2 +6], new_column_names_unc))
        Kd = Kd.rename(columns=column_mapping_unc)
        new_column_names_SE = ["kd" + str(wavelength) + "_se" for wavelength in wavelengths]
        column_mapping_SE = dict(zip(Kd.columns[2*len(wavelengths) + 6: len(wavelengths) * 3 + 6], new_column_names_SE))
        Kd = Kd.rename(columns=column_mapping_SE)
        new_column_names_bin = ["kd" + str(wavelength) + "_bincount" for wavelength in wavelengths]
        column_mapping_bin = dict(zip(Kd.columns[3*len(wavelengths) + 6: len(wavelengths) * 4 + 6], new_column_names_bin))
        Kd = Kd.rename(columns=column_mapping_bin)

        new_column_names = ["ed0" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Ed0.columns[5:len(wavelengths)+5], new_column_names))
        Ed0 = Ed0.rename(columns=column_mapping)
        new_column_names_unc = ["ed0" + str(wavelength) + "_unc" for wavelength in wavelengths]
        column_mapping_unc = dict(zip(Ed0.columns[len(wavelengths)+6 :len(wavelengths)*2 +6], new_column_names_unc))
        Ed0 = Ed0.rename(columns=column_mapping_unc)


     # Calculate the median and standard deviation across the bootstrap samples
        median_luf_kd, std_luf_kd, median_Ed0, std_Ed0 = bootstrap_fit_klu_depth(new_Ed, Speed, n_iterations=100, fit_method='iterative')
        result = Function_KD.fit_klu(new_Ed,  fit_method='iterative', wl_interp_method='None', smooth_method='None',  only_continuous_obs=False)
        result_Kd = result['Kl']
        SE_Kd = result['Luf_sd']/np.sqrt(result['data_count'])
        if ~(np.isnan(median_luf_kd)).all():
            result_Kd = result_Kd.mask(result_Kd < 0, np.nan)
            # Calculate uncertainties
            Kd_uncertainty = std_luf_kd/np.sqrt(100)  # Take standard error = std/sqt(100)
        else:
            Kd_uncertainty = pd.Series([np.nan] * len(wavelengths))
            std_Ed0 = pd.Series([np.nan] * len(wavelengths))

        data_dict_K ={
            'profile': int(current_cycle),
            'date': Ed_profile.date[0],
            'time': Ed_profile.time[0],
            'lon': Ed_profile.lon[0].round(5),
            'lat': Ed_profile.lat[0].round(5),
            'quality': Ed_profile['quality'][0]
        }

        data_dict_K.update(dict(zip(Kd.columns[6: len(wavelengths) +6],result_Kd)))
        data_dict_K.update(dict(zip(Kd.columns[len(wavelengths) +6: len(wavelengths)*2 +6], Kd_uncertainty.values.reshape(-1).astype(np.float32))))
        data_dict_K.update(dict(zip(Kd.columns[len(wavelengths)*2 +6: len(wavelengths)*3 +6],SE_Kd)))
        data_dict_K.update(dict(zip(Kd.columns[len(wavelengths)*3 +6 : len(wavelengths)*4 +6], result['data_count'])))

        data_Kd.append(data_dict_K)

        data_dict ={
             'profile': int(current_cycle),
            'date': Ed_profile.date[0],
            'time': Ed_profile.time[0],
            'lon': Ed_profile.lon[0].round(5),
            'lat': Ed_profile.lat[0].round(5),
            'quality':  Ed_profile['quality'][0]
        }

        data_dict.update(dict(zip(Ed0.columns[5: len(wavelengths) +5], median_Ed0)))
        data_dict.update(dict(zip(Ed0.columns[len(wavelengths) + 6: len(wavelengths) * 2 + 6],
                                    std_Ed0.values.reshape(-1).astype(np.float32))))

        data_Ed0.append(data_dict)

        # save csv file
        # Move 'date' and 'time' columns to the front
        columns = ['date', 'time','depth'] + [col for col in Ed_profile.columns if col not in ['date', 'time','depth']]
        Ed_profile = Ed_profile[columns]

        #Create the .csv file
        # fileN ='PVST_VDIUP-Argo-Kd_'+ wmo + '_' + current_cycle + '_Ed' +'_raw'
        path = os.path.join(Processed_profiles, wmo )
        # Ed_profile.to_csv(os.path.join(path, fileN +'_raw.csv'), index=False)
        # #Create the .sb file
        # sb.format_to_seabass(Ed_profile, metadata, fileN, path, comments, missing_value_placeholder= '-9999', delimiter= 'comma')

        Ed_with_station = Ed_profile.copy()
        Ed_with_station['profile'] = current_cycle
        Ed_physic = pd.concat([Ed_physic, Ed_with_station])

    Kd = pd.DataFrame(data_Kd)
    Ed0 = pd.DataFrame(data_Ed0)

    flags_df = data_flags

    # Load the watercoeff file
    watercoeff = pd.read_csv('watercoeff.csv')

    # Iterate through the Kd DataFrame and update values

    for col in Kd.columns:
        # Extract the wavelength from the column name
        match = re.search(r'kd(\d+)', col)
        if match:
            wavelength = int(match.group(1))
            if wavelength > 700:
                # Compare Kd values with the aw column in watercoeff
                aw_value = watercoeff.loc[watercoeff['lambda'] == wavelength, 'aw'].values[0]
                unc_col = f'kd{wavelength}.0_unc'
                ed0_col = f'ed0{wavelength}.0'
                ed0_unc_col = f'ed0{wavelength}.0_unc'

                def replace_if_less(row):
                    if row[col] < aw_value:
                        return np.nan
                    else:
                        return row[col]

                    # Apply the condition to Kd dataset

                Kd[col] = Kd.apply(lambda row: replace_if_less(row), axis=1)
                Kd[unc_col] = Kd.apply(lambda row: replace_if_less(row), axis=1)

                # Apply the condition to Ed0 dataset
                Ed0[ed0_col] = Kd.apply(lambda row: replace_if_less(row), axis=1)
                Ed0[ed0_unc_col] = Kd.apply(lambda row: replace_if_less(row), axis=1)

    if not Kd.empty:
         #   Kd = Kd.round(4)
            Kd.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Kd.csv'), index=False)
            print(f'Kd file for float {wmo} was created')
                 # Group the Kd DataFrame by month
            Kd['year_month'] = pd.to_datetime(Kd['date']).dt.strftime('%Y%m')
            # Iterate over each group and create a SeaBASS file
            for year_month, group in Kd.groupby('year_month'):
                group = group.drop(columns=['year_month'])
                sb.format_to_seabass(group, metadata, f'PVST_VDIUP-Argo-Kd_{wmo}_{year_month}_R0', path, comments,
                                     missing_value_placeholder='-9999', delimiter='comma')
            Ed0.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed0.csv'), index=False)
            print(f'Ed0 file for float {wmo} was created')
            Ed_physic.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed.csv'), index=False)
            print(f'Ed file for float {wmo} was created')

    # Filter out columns in Kd that contain '_se' or '_bincount'
            plot_ed_profiles(df=Ed_physic, wmo=wmo, kd_df=Kd, wv_target=[490, 555, 660], wv_og=wavelengths, ed0=Ed0,flags_df = flags_df,
                   depth_col='depth')
#
QC_5wv = pd.DataFrame(Check_5wv)
#
# # SAve to csv
QC_5wv.to_csv(os.path.join(Processed_profiles, 'QC_5wv.csv'), index=False)
combined_QC_5wv.to_csv(os.path.join(Processed_profiles, 'combined_QC_5wv.csv'), index=False)
# %% QUALITY CONTROL OF ALL KD PROFILES FROM A SINGLE FLOAT

# Load the Kd profiles
Kd = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Kd.csv')))
Ed0 = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed0.csv')))
Ed_all = pd.read_csv(os.path.join(Processed_profiles, wmo, wmo + '_Ed.csv'))

def extracted_wavelengths(Ed, pattern='ed'):
    # Extract wavelengths from the Kd DataFrame
    columns = [col for col in Ed.columns if col.startswith(pattern) and 'unc' not in col]

    wavelengths = []
    for col in columns:
        match = re.search(pattern + r'(\d+\.?\d*)', col)
        if match:
            wavelengths.append(float(match.group(1)))
        else:
            wavelengths.append(np.nan)
    wavelengths = np.array(wavelengths)
    return wavelengths

wavelengths = extracted_wavelengths(Ed_all, pattern='ed')

plot_ed_profiles(df = Ed_all,wmo =  wmo, kd_df =Kd, wv_target = [490, 550, 660], wv_og=wavelengths, ed0= Ed0, depth_col='depth')

# %%
# Interactive figure to see if all Kd are really good and assign them a bad flag if not

kd_columns = [col for col in Kd.columns if col.startswith('kd') and 'unc' not in col]
# Calculate the mean Kd and the mean Ed0 for each wavelength
mean_kd = Kd[kd_columns].mean()
mean_ed0 = Ed0[[col.replace('kd', 'ed0') for col in kd_columns]].mean()

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['quality'] ==0]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.profile}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
plt.ylim([0, 1])
fig.suptitle('Good Kd profiles for float ' + wmo)
plt.show()
fig.savefig(os.path.join(Processed_profiles, wmo,wmo + '_Kd_good.png'))

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['quality'] ==2]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.profile}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
plt.ylim([0, 3])
fig.suptitle('BAD Kd profiles for float ' + wmo)
plt.show()
fig.savefig(os.path.join(Processed_profiles, wmo,wmo + '_Kd_bad.png'))


# %% Try and do figure for paper
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Filter the DataFrame to get rows where profile == 8 and depth is <= 100 meters
filtered_df = Ed_all[(Ed_all['profile'] == 13) & (Ed_all['depth'] <= 100)]

# Extract the columns that start with 'ed'
ed_columns = [col for col in filtered_df.columns if col.startswith('ed')]

# Plot the ed values as a function of wavelength for each row
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

# Top plot: Ed values as a function of wavelength
ax1 = fig.add_subplot(gs[0])
colors = cm.viridis(np.linspace(0, 1, len(filtered_df)))
for idx, (row, color) in enumerate(zip(filtered_df.iterrows(), colors)):
    ax1.plot(wavelengths, row[1][ed_columns], color=color, alpha=0.7)

# Add labels and title
ax1.set_xlabel('Wavelength (nm)',fontsize=16)
ax1.set_ylabel('Ed values',fontsize=16)
ax1.set_title('Float ' + wmo+ ' Ed Spectra for profile 10',fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)


# Create a color bar
norm = mcolors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Depth (m)',fontsize = 16)
cbar.ax.tick_params(labelsize=12)

# Bottom plots: Depth as a function of Ed for specific wavelengths
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1])

# Define the specific wavelengths and colors
specific_wavelengths = [381.0, 441.0, 487.0, 554.0, 621.0]
plot_colors = ['purple','indigo', 'lightblue', 'green', 'red']

for i, (wavelength, plot_color) in enumerate(zip(specific_wavelengths, plot_colors)):
    ax = fig.add_subplot(gs_bottom[i])
    ed_column = f'ed{wavelength}'
    for idx, row in filtered_df.iterrows():
        ax.scatter(row[ed_column], row['depth'], color=plot_color, alpha=0.7, marker='o')
    if i == 0:
        ax.set_ylabel('Depth (m)', fontsize=16)
    else:
        ax.tick_params(axis='y', labelleft=False)  # Disable y-axis labels but keep tick marks
    ax.set_xlabel(f'Ed {wavelength} (nm)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.invert_yaxis()  # Reverse the depth axis



plt.tight_layout()
plt.show()