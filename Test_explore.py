import glob
import os
import re
import sys

import cmocean
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve')
import Toolbox_RAMSES as tools
import matplotlib.pyplot as plt
import Function_KD
import gsw
import Organelli_QC
import mplcursors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
root = '/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve'
Processed_profiles = '/Users/charlotte.begouen/Documents/Hyperspectral_floats_Herve/Outputs'

# %% Get WMO - automatically go through the list of all floats
wmos = []

for item in os.listdir(root):
    full_path = os.path.join(root, item)
    # Check if the item is a directory and matches the regular expression
    if os.path.isdir(full_path) and re.match("^\d+$", item):
        wmos.append(item)

wmo = '6990503'

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
def plot_ed_profiles(df, kd_df, wv_target, wv_og, ed0, depth_col='depth'):
    # Define the path to the figure
    wmo = df['CRUISE'].values[0]

    # Extract the ED columns from the DataFrame
    ed_columns = [col for col in df.columns if col.startswith('ED')]
    kd_columns = [col for col in kd_df.columns if col.startswith('KD')]
    ed0_columns = [col for col in ed0.columns if col.startswith('Ed0')]

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
    for cycle in df['CYCLE'].unique():
        # Define the path to the figure
        figure_path = os.path.join(Processed_profiles, str(df['CRUISE'].values[0]), f"{df[df['CYCLE'] == cycle]['WMO'].iloc[0]}_fig.png")

        # Check if the figure already exists
        if not os.path.exists(figure_path):
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
            flag = kd_df[kd_df['CYCLE'] == cycle]['FLAGS'].values[0][0]
            if flag.startswith('GOOD'):
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
            for inner_cycle in df['CYCLE'].unique():
                if inner_cycle != cycle:
                    ax1.plot(wv_og, ed0[ed0['CYCLE'] == inner_cycle][ed0_columns].values[0], color='lightgrey')
                    ax2.plot(wv_og, kd_df[kd_df['CYCLE'] == inner_cycle][kd_columns].values[0], color='lightgrey')

            ax1.plot(wv_og, ed0[ed0['CYCLE'] == cycle][ed0_columns].values[0], color='blue', linewidth=2)
            ax2.plot(wv_og, kd_df[kd_df['CYCLE'] == cycle][kd_columns].values[0], color='blue', linewidth=2)

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Ed(0-) Values')
            ax1.set_title('Hyperspectral Ed(0-)')
            ax2.set_ylim([0.90 * min(ed0[ed0['CYCLE'] == cycle][ed0_columns].values[0]),
                          1.05 * max(ed0[ed0['CYCLE'] == cycle][ed0_columns].values[0])])
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Kd Values')
            ax2.set_title('Hyperspectral Kd')
            ax2.set_ylim([0.90* min(kd_df[kd_df['CYCLE'] == cycle][kd_columns].values[0]),
                          1.05 * max(kd_df[kd_df['CYCLE'] == cycle][kd_columns].values[0])])

            # Plot the ED profiles
            colors = ['blue', 'green', 'red']  # Define a list of colors

            for idx, (ed_col, kd_col, ed0_col) in enumerate(
                    zip(closest_columns, closest_kd_columns, closest_0_columns)):

                if idx == 0:
                    ax = ax3
                elif idx == 1:
                    ax = ax4
                elif idx == 2:
                    ax = ax5

                kd_value = kd_df[kd_df['CYCLE'] == cycle][kd_col].values[0]
                ed0_s = ed0[ed0['CYCLE'] == cycle][ed0_col].values[0]
                df_filtered_flags = df[(df['CYCLE'] == cycle) & (df['FLAGS'] == 0)]
                ax.scatter(df[df['CYCLE'] == cycle][ed_col], df[df['CYCLE'] == cycle][depth_col],
                           label=f'{ed_col}', c=colors[idx], alpha= 0.3 ,marker =  'x')
                ax.scatter(df_filtered_flags[ed_col],df_filtered_flags[depth_col],
                           label=f'{ed_col}', c = colors[idx])

                new_depth_values = np.linspace(df[df['CYCLE'] == cycle][depth_col].min(), 50, len(df[depth_col]))
                ed_predicted = ed0_s * np.exp(-kd_value * new_depth_values)
                ax.plot(ed_predicted, new_depth_values, '--', label=f'Predicted ED from {kd_col}', color=colors[idx])
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
            Kd = pd.DataFrame(
                columns=['CRUISE', 'CYCLE', 'WMO', 'TIME', 'LON', 'LAT'])
            print(f'Retrieval failed for float {wmo} Kd. Creating new dataframe...')
            for i in range(70):
                Kd[i] = pd.NA

        try:
            Ed0 = pd.read_csv(os.path.join(Processed_profiles, wmo, (wmo + '_Ed0.csv')))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            Ed0 = pd.DataFrame(
                columns=['CRUISE', 'CYCLE', 'WMO', 'TIME', 'LON', 'LAT'])
            print(f'Retrieval failed for float {wmo} Ed0. Creating new dataframe...')
            for i in range(70):
                Ed0[i] = pd.NA

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

        current_cycle = re.search(r"_([0-9]{3})_", filename).group(1)
        if current_cycle in processed_cycles and int(current_cycle) in Kd['CYCLE'].values and int(current_cycle) in \
                Ed_physic['CYCLE']:
            print(f'Profile {current_cycle} already processed for float {wmo}. Skipping...')
            continue

        if (Kd['CYCLE'] == int(current_cycle)).any() and (Ed_physic['CYCLE'] == int(current_cycle)).any() and (
                Ed0['CYCLE'] == int(current_cycle)).any():
            print(f'Profile {current_cycle} already processed for float {wmo}. Skipping...')
            continue

        # if iteration_count >= max_iterations:
        #     break  # Exit the loop if maximum iterations reached
        #
        # iteration_count += 1
        # print(os.path.basename(filename))
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

        if not (len(ed_n_prof) > 0 and len(lu_n_prof) > 0):
            print('skip')
            continue

        if np.isnan(data.RAW_DOWNWELLING_IRRADIANCE.values).all() or np.isnan(data.RAW_UPWELLING_RADIANCE.values).all():
            print('skip')
            continue

        Ed_n_prof, Lu_n_prof = ed_n_prof[0][0], lu_n_prof[0][0]

        # Read RAMSES Data
        Ed_physic_profile, Lu_physic_profile = tools.format_ramses(
            filename, meta_filename, cals[(cals['rad'] == 'Ed') & (cals['wmo'] == wmo)]['calibration_file'].iloc[0],
            cals[(cals['rad'] == 'Lu') & (cals['wmo'] == wmo)]['calibration_file'].iloc[0], Ed_n_prof, Lu_n_prof)

        # Read Meta Data
        basename = os.path.basename(filename)
        metadata_ed = pd.DataFrame({'CRUISE': [basename[1:8]] * Ed_physic_profile.shape[0],
                                    'CYCLE': [int(basename[-10:-7])] * Ed_physic_profile.shape[0],
                                    'WMO': [basename[1:-7]] * Ed_physic_profile.shape[0],
                                    'TIME': [data.JULD.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0],
                                    'LON': [data.LONGITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0],
                                    'LAT': [data.LATITUDE.sel(N_PROF=Ed_n_prof).values] * Ed_physic_profile.shape[0]})

        pres_ed = data.PRES.sel(N_PROF=Ed_n_prof).values[0:Ed_physic_profile.shape[0]]

        metadata_Lu = pd.DataFrame({'CRUISE': [basename[1:8]] * Lu_physic_profile.shape[0],
                                    'CYCLE': [int(basename[-10:-7])] * Lu_physic_profile.shape[0],
                                    'WMO': [basename[1:-7]] * Lu_physic_profile.shape[0],
                                    'TIME': [data.JULD.sel(N_PROF=Lu_n_prof).values] * Lu_physic_profile.shape[0],
                                    'M_TIME': data.MTIME.sel(N_PROF=Lu_n_prof).values[0: Lu_physic_profile.shape[0]],
                                    'LON': [data.LONGITUDE.sel(N_PROF=Lu_n_prof).values] * Lu_physic_profile.shape[0],
                                    'LAT': [data.LATITUDE.sel(N_PROF=Lu_n_prof).values] * Lu_physic_profile.shape[0]})

        pres_Lu = data.PRES.sel(N_PROF=Lu_n_prof).values[0:Lu_physic_profile.shape[0]]

        # Correct to the right timestamp
        metadata_ed.TIME = metadata_ed.TIME - data.MTIME.sel(N_PROF=Ed_n_prof).values[0: Ed_physic_profile.shape[0]]
        metadata_Lu.TIME = metadata_Lu.TIME - data.MTIME.sel(N_PROF=Lu_n_prof).values[0: Lu_physic_profile.shape[0]]

        # Calculate Speed with depth
        metadata_ed['SPEED'] = np.nan
        metadata_Lu['SPEED'] = np.nan
        for i in range(1, len(metadata_ed)):
            dt = metadata_ed.TIME[i] - metadata_ed.TIME[i - 1]
            dpres = pres_ed[i] - pres_ed[i - 1]
            metadata_ed.loc[i, "SPEED"] = dpres / dt.seconds  # in m/s
        for i in range(1, len(metadata_Lu)):
            dt = metadata_Lu.TIME[i] - metadata_Lu.TIME[i - 1]
            dpres = pres_Lu[i] - pres_Lu[i - 1]
            metadata_Lu.loc[i, "SPEED"] = dpres / dt.seconds  # in m/s

        # Compute difference of 2 seconds for the depth
        delta_depth = metadata_ed.SPEED * 2
        metadata_ed['DEPTH'] = pres_ed - delta_depth
        metadata_ed.loc[0, 'DEPTH'] = pres_ed[0] - delta_depth[1]
        delta_depth = metadata_Lu.SPEED * 2
        metadata_Lu['DEPTH'] = pres_Lu - delta_depth
        metadata_Lu.loc[0, 'DEPTH'] = pres_Lu[0] - delta_depth[1]

        # Interpolate the Base data so it is at the same depth as the optics data : No extrapolation is done
        interp_temp = np.interp(metadata_ed.DEPTH, data_base.PRES.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]],
                                data_base.TEMP.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        interp_psal = np.interp(metadata_ed.DEPTH, data_base.PRES.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]],
                                data_base.PSAL.sel(N_PROF=0).values[0:Ed_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        metadata_ed['TEMP'] = interp_temp
        metadata_ed['PSAL'] = interp_psal

        interp_temp = np.interp(metadata_Lu.DEPTH,
                                data_base.PRES.sel(N_PROF=0).values[0:Lu_physic_profile.shape[0]],
                                data_base.TEMP.sel(N_PROF=0).values[0:Lu_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        interp_psal = np.interp(metadata_Lu.DEPTH,
                                data_base.PRES.sel(N_PROF=0).values[0:Lu_physic_profile.shape[0]],
                                data_base.PSAL.sel(N_PROF=0).values[0:Lu_physic_profile.shape[0]], left=np.nan,
                                right=np.nan)
        metadata_Lu['TEMP'] = interp_temp
        metadata_Lu['PSAL'] = interp_psal

        CT = gsw.conversions.CT_from_t(metadata_ed.PSAL, metadata_ed.TEMP, metadata_ed.DEPTH)
        metadata_ed['DENS'] = gsw.rho(metadata_ed.PSAL, CT, metadata_ed.DEPTH)

        CT = gsw.conversions.CT_from_t(metadata_Lu.PSAL, metadata_Lu.TEMP, metadata_Lu.DEPTH)
        metadata_Lu['DENS'] = gsw.rho(metadata_Lu.PSAL, CT, metadata_Lu.DEPTH)

        MLD = zmld_boyer(metadata_ed.PSAL, metadata_ed.TEMP, metadata_ed.DEPTH)

        # Concatenate
        Ed_profile = pd.concat([metadata_ed, Ed_physic_profile], axis=1)
        Lu_profile = pd.concat([metadata_Lu, Lu_physic_profile], axis=1)

        # Extract wavelength to format for Kd function and rename columns
        wavelengths = [col for col in Ed_profile.columns if isinstance(col, (int, float))]

        # Organelli QC
        [f1, f3, g5, p_fit, qc_wv]  = Organelli_QC.organelli16_qc(Ed_profile, lat=Ed_profile.LAT[0],
                                                          lon=Ed_profile.LON[0],qc_wls=   [380,400,490,500], step2_r2=0.995, step3_r2=0.998,
                                                          skip_meta_tests=False, skip_dark_test=True)

        if f1 == False:
            print(f"Cycle {current_cycle} failed QC for {g5}: Careful proceeding")
            Ed_profile['FLAGS'] = 2  # Bad flags, don't use
        else:
            Ed_profile['FLAGS'] = f3

            # Plot the fitted polynomial - OPTIONAL

            for flag_value, color in zip([0, 1, 2], ['green', 'orange', 'red']):
                sel = Ed_profile['FLAGS'] == flag_value
                plt.scatter(Ed_profile[489.0][sel],-Ed_profile['DEPTH'][sel],  label=f'Flag: {flag_value}', color =color)

            plt.plot(np.exp(p_fit['fitted_values']),- p_fit['depths'], color='black', label='2nd polynomial fit')

            plt.ylabel('Depth (m)')
            plt.xlabel(' Ed(489)')
            plt.title(title + ': Organelli QC at 489nm')
            plt.legend()
            plt.ylim([ -100 , 0])
            plt.show()

        for col in wavelengths:
            Ed_profile.rename(columns={col: 'ED' + str(col)}, inplace=True)

        # Select rows to use
        idx_end = len(Ed_profile) - 1
        lu_columns = [col for col in Ed_profile.columns if col.startswith(('ED', 'LU'))]


        # Plot hyperspectral Ed to see if has expected shape at all wavelengths (Only if passed QC)
        #  if f1 == True :
        #      cmap = plt.get_cmap('viridis')
        #      normalize = mcolors.Normalize(0,100)
        #      for N, row in Ed_profile.iterrows():
        #          if row['FLAGS'] == 0: # Good profiles only
        #              plt.plot(wavelengths, row[lu_columns],  color = cmap(normalize(Ed_profile['DEPTH'][N])))
        #     # Add labels and title
        # plt.xlabel( 'wavelength (nm)')
        # plt.ylabel('ED Values')
        # plt.title(title + ': Hyperspectral Ed')
        # plt.show()

        ###### Create Kd document ######

        if f1 == True:
            flag_idx = Ed_profile['FLAGS'].isin([0, 1])  # Omit rows with flags 2 (BAD)
        else:
            flag_idx = Ed_profile['FLAGS'].isin([0, 1, 2])

        new_Ed = Ed_profile.loc[flag_idx, ['TIME', 'DEPTH'] + lu_columns].copy()
        new_Ed = new_Ed.rename(columns={'TIME': 'datetime', 'DEPTH': 'depth'})

        # Add to global table of the float
        new_column_names = ["KD" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Kd.columns[6:77], new_column_names))
        Kd = Kd.rename(columns=column_mapping)

        new_column_names = ["Ed0" + str(wavelength) for wavelength in wavelengths]
        column_mapping = dict(zip(Ed0.columns[6:77], new_column_names))
        Ed0 = Ed0.rename(columns=column_mapping)

        kd_res = Function_KD.fit_klu(new_Ed, fit_method='iterative', wl_interp_method='None', smooth_method='None',
                                     only_continuous_obs=False)
        data_Kd.append({
            'CYCLE': Ed_profile.CYCLE[0],
            'WMO': Ed_profile.WMO[0],
            'TIME': Ed_profile.TIME[0],
            'LON': Ed_profile.LON[0],
            'LAT': Ed_profile.LAT[0],
            'FLAGS': [g5],
            **dict(zip(Kd.columns[6: 76], kd_res.Kl.values.reshape(-1).astype(np.float32)))
        })

        data_Ed0.append({
            'CRUISE': Ed_profile.CRUISE[0],
            'CYCLE': Ed_profile.CYCLE[0],
            'WMO': Ed_profile.WMO[0],
            'TIME': Ed_profile.TIME[0],
            'LON': Ed_profile.LON[0],
            'LAT': Ed_profile.LAT[0],
            'FLAGS': [g5],
            **dict(zip(Ed0.columns[6: 76], kd_res.Luf.values.reshape(-1).astype(np.float32)))
        })
        # Plot the profile with the Kd values for 4 wavelengths.
        # plot_ed_profiles(new_Ed, Kd.loc[idx].to_frame().T, [380,490, 700], wavelengths, Ed0.loc[idx].to_frame().T, flag_idx, title, depth_col='depth')

        # save csv file
        Ed_profile.to_csv(os.path.join(Processed_profiles, wmo, wmo + '_' + current_cycle + '_Ed.csv'), index=False)
        print(f" Ed profile for float {wmo} Cycle {current_cycle} was created")

        Ed_physic = pd.concat([Ed_physic, Ed_profile])

    Kd = pd.DataFrame(data_Kd)
    Ed0 = pd.DataFrame(data_Ed0)

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

def extracted_wavelengths(Kd, pattern='KD'):
    # Extract wavelengths from the Kd DataFrame
    columns = [col for col in Kd.columns if col.startswith('KD')]

    wavelengths = []
    for col in columns:
        match = re.search(pattern + r'(\d+\.?\d*)', col)
        if match:
            wavelengths.append(float(match.group(1)))
        else:
            wavelengths.append(np.nan)
    wavelengths = np.array(wavelengths)
    return wavelengths


wavelengths = extracted_wavelengths(Kd, pattern='KD')

plot_ed_profiles(Ed_all, Kd, [490, 550, 660], wavelengths, Ed0, depth_col='DEPTH')

kd_columns = [col for col in Kd.columns if col.startswith('KD')]

# Calculate the mean Kd and the mean Ed0 for each wavelength
mean_kd = Kd[kd_columns].mean()
mean_ed0 = Kd[[col.replace('KD', 'Ed0') for col in kd_columns]].mean()

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['FLAGS'].astype(str).str.contains('GOOD')]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.CYCLE}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
plt.ylim([0, 2])
cursor = mplcursors.cursor(lines, hover=True)


@cursor.connect("add")
def on_add(sel):
    line_idx = lines.index(sel.artist)
    cycle = filtered_data.iloc[line_idx]['CYCLE']
    sel.annotation.set(text=f'Cycle {cycle}', position=(0, 20), anncoords="offset points")
    sel.annotation.xy = (sel.target[0], sel.target[1])


fig.suptitle('Good Kd profiles for float ' + wmo)
plt.show()

fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
# Plot each row in the filtered DataFrame
filtered_data = Kd[Kd['FLAGS'].astype(str).str.contains('BAD')]
lines = []
for idx, row in filtered_data.iterrows():
    line, = ax.plot(wavelengths, row[kd_columns], label=f'Cycle {row.CYCLE}')
    lines.append(line)
# Set number of ticks
num_ticks = 10
plt.xticks(np.linspace(min(wavelengths), max(wavelengths), num_ticks))
cursor = mplcursors.cursor(lines, hover=True)

plt.ylim([0, 0.5])
@cursor.connect("add")
def on_add(sel):
    line_idx = lines.index(sel.artist)
    cycle = filtered_data.iloc[line_idx]['CYCLE']
    sel.annotation.set(text=f'Cycle {cycle}', position=(0, 20), anncoords="offset points")
    sel.annotation.xy = (sel.target[0], sel.target[1])


fig.suptitle('BAD Kd profiles for float ' + wmo)
plt.show()

# %%
