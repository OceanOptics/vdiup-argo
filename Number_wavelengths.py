"""
This computes a QC based on the same criteria as the  Andres & Begouen Demeaux et al., 2025 paper
 BUT on an increasing number of wavelengths to compute what happens in Figure 10.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pvlib

def find_closest_wavelengths(targets, available_wavelengths):
    closest_wavelengths = []
    for target in targets:
        closest = min(available_wavelengths, key=lambda x: abs(x - target))
        closest_wavelengths.append(closest)
    return closest_wavelengths

def QC_all_wavelengths(tab_wmo, wmo, sensor, nonshaded_min, to_save,columns_to_skip):
    """
    Function to compute a high-level Quality Control (QC) on BGC-Argo float hyperspectral data (Ed and Lu).
    This version processes all available wavelengths dynamically.

    Parameters
    ----------
    tab_wmo : pandas.DataFrame
        Table with all the Ed or Lu profiles from one float.
        Columns are: CRUISE CYCLE WMO TIME LON LAT PRES_FLOAT Post_Pres lambda1 lambda2 lambda3 ... lambda70.
    wmo : str
        WMO identification number of the current float.
    to_save : str
        Path where the QC csv file will be saved.
        The file name will be: {wmo}_QC.csv

    Returns
    -------
    tab_wmo_qualified: pandas.DataFrame
        Table with all the Ed or Lu profiles from one float, with QC applied to all wavelengths.
    """
    tab_wmo_qualified = pd.DataFrame()

    # Extract all wavelength columns dynamically (assuming they start from column index 8)
    wavelength_columns = pd.to_numeric(tab_wmo.columns[columns_to_skip:])
    indexes = list(range(columns_to_skip, len(tab_wmo.columns)))
    indexes_names = [str(w) for w in wavelength_columns]

    # Apply QC profile by profile
    for cyc in tab_wmo.CYCLE.unique():
        # Select the current cycle in profile data
        profile_cyc0 = tab_wmo[tab_wmo.CYCLE == cyc]
        if len(profile_cyc0) < 5:  # Skip profiles with fewer than 5 points
            continue

        # Clean depth vector
        profile_cyc = profile_cyc0.drop_duplicates(subset='PRES_FLOAT', keep='first', ignore_index=True)
        profile_cyc_qc = profile_cyc.copy()

        ### START QC PROCEDURE
        ### STEP 0: Test if night profile
        solar_angle = pvlib.solarposition.get_solarposition(
            time=pd.DatetimeIndex(data=[profile_cyc.TIME.iloc[0]], tz='utc'),
            longitude=profile_cyc.LON.iloc[0],
            latitude=profile_cyc.LAT.iloc[0]
        )

        if solar_angle['zenith'].values.mean() < 2 or solar_angle['zenith'].values.mean() > 178:
            for i, ind in enumerate(indexes):  # Attribute type 3 & save QC in a global table
                profile_cyc_qc[f'Type_{indexes_names[i]}'] = [3] * len(profile_cyc_qc)
            tab_wmo_qualified = pd.concat([tab_wmo_qualified, profile_cyc_qc], axis=0)
            continue

        ## Filter for tilt > 5° (following IOCCG reccomandations)
        profile_cyc = profile_cyc[(profile_cyc.Post_Tilt < 5)]

        ### only for Lu: Save saa and shading for later
        # if (sensor=='Lu'):
        #     saa = solar_angle['azimuth'].values.mean()
        #     angle = (saa+profile_cyc.heading)%360

        # Iterate over all wavelengths
        for i, ind in enumerate(indexes):
            ## STEP 1: Dark identification with Shapiro test
            shapiro_tab = pd.DataFrame(columns=['depth', 'pvalue'])
            for j, z in enumerate(profile_cyc.PRES_FLOAT.iloc[:-4]):
                shapiro_test = stats.shapiro(profile_cyc[profile_cyc.PRES_FLOAT > z].iloc[:, ind])
                shapiro_tab.loc[j, :] = [z, shapiro_test[1]]

            shapiro_tab.depth, shapiro_tab.pvalue = pd.to_numeric(shapiro_tab.depth), pd.to_numeric(shapiro_tab.pvalue)
            p0_index = (shapiro_tab.pvalue > 10 ** (-5)).idxmax() if (shapiro_tab.pvalue > 10 ** (-5)).any() else 0
            if p0_index == 0:
                profile_cyc_qc['Type_{}'.format(indexes_names[i])] = [3] * len(profile_cyc_qc)
                continue
            else:
                z_dark = shapiro_tab.loc[p0_index, 'depth']

            ### Profile points <= 5
            if len(profile_cyc[profile_cyc.PRES_FLOAT < z_dark].iloc[:, ind]) <= 5:  # Attribute type 3
                profile_cyc_qc[f'Type_{indexes_names[i]}'] = [3] * len(profile_cyc_qc)
                continue

            else:
                ### only for Lu: Count for the proportion of shaded data on the whole profile (even dark data)
                if (sensor == 'Lu'):
                    n_tot = len(profile_cyc[(profile_cyc.PRES_FLOAT < z_dark)].iloc[:, ind])
                    # nonshaded_percent = len(profile_cyc[(profile_cyc.PRES_FLOAT<z_dark)&(angle>135)&(angle<315)].iloc[:,ind]) / n_tot

                    # if nonshaded_percent*100 < nonshaded_min:
                    #     profile_cyc_qc['Type_{}'.format(indexes_names[i])] = [3]*len(profile_cyc_qc)
                    #     continue

                ### STEP 2: Nonlinear fit after dark removal
                log = np.log(profile_cyc[profile_cyc.PRES_FLOAT < z_dark].iloc[:, ind])
                z_fit = profile_cyc.loc[profile_cyc.PRES_FLOAT < z_dark, 'PRES_FLOAT']
                p = np.polyfit(z_fit, log, 4)
                fit = np.polyval(p, z_fit)
                r2 = round(np.corrcoef(log, fit)[0, 1] ** 2, 3)

                if r2 < 0.995:  # Attribute type 3
                    profile_cyc_qc[f'Type_{indexes_names[i]}'] = [3] * len(profile_cyc_qc)
                    continue

                ### Compute residuals
                qc_tab = pd.DataFrame({
                    'depth': z_fit,
                    'data': profile_cyc[profile_cyc.PRES_FLOAT < z_dark].iloc[:, ind],
                    'fit': fit,
                    'residuals': np.abs(log - fit),
                    'r2': r2,
                    'flag': [np.nan] * len(z_fit)
                })

                # Flag datapoints depending on residuals
                qc_tab.loc[
                    (np.abs(qc_tab.residuals) > qc_tab.residuals.mean() + 1 * qc_tab.residuals.std()), 'flag'] = 2
                qc_tab.loc[
                    (np.abs(qc_tab.residuals) > qc_tab.residuals.mean() + 2 * qc_tab.residuals.std()), 'flag'] = 3

                ### Nonlinear fit after removal of flag 3 data
                log2 = np.log(qc_tab.loc[qc_tab.flag != 3, 'data'])
                z_fit2 = qc_tab.loc[qc_tab.flag != 3, 'depth']
                p2 = np.polyfit(z_fit2, log2, 4)
                fit2 = np.polyval(p2, z_fit2)
                r22 = round(np.corrcoef(log2, fit2)[0, 1] ** 2, 3)

                # Save QC for the wavelength in a profile table
                profile_cyc_qc[f'Type_{indexes_names[i]}'] = np.nan

                # Determine boundaries (bornes) based on the wavelength
                wavelength = float(indexes_names[i])  # Convert the wavelength string to a float
                if wavelength < 600:
                    bornes = [0.996, 0.998]
                else:
                    bornes = [0.995, 0.998]

                # Assign QC types based on r22 and the boundaries
                if r22 <= bornes[0]:
                    profile_cyc_qc[f'Type_{indexes_names[i]}'] = 3
                elif bornes[0] < r22 <= bornes[1]:
                    profile_cyc_qc[f'Type_{indexes_names[i]}'] = 2
                else:
                    profile_cyc_qc[f'Type_{indexes_names[i]}'] = 1

        # Save QC in a global table
        tab_wmo_qualified = pd.concat([tab_wmo_qualified, profile_cyc_qc], axis=0)
    return tab_wmo_qualified

def compute_global_type(tab_wmo_qualified):
     # Extract type columns (who have the QC flags) for wavelengths <= 650 nm
     type_columns_hyper = [col for col in tab_wmo_qualified.columns if col.startswith('Type_')]
     filtered_type_columns = [col for col in type_columns_hyper if float(col.split('_')[1]) <= 650]

     # Count occurrences of each type
     type_counts = tab_wmo_qualified[filtered_type_columns].apply(pd.Series.value_counts, axis=1).fillna(0)
     total_wavelengths = type_counts.sum(axis=1)

     # Calculate percentages
     pct_type1 = (type_counts.get(1, 0) / total_wavelengths) * 100
     pct_type3 = (type_counts.get(3, 0) / total_wavelengths) * 100

     # Assign Global_Type based on conditions
     global_type = pd.Series(2, index=tab_wmo_qualified.index)  # Default to "Questionable" (2)
     global_type[(pct_type1 < 40) & (pct_type3 > 20)] = 3  # "Bad"
     global_type[(pct_type1 >= 40) & (pct_type3 <= 20)] = 2  # "Questionable"
     global_type[(pct_type1 >= 80) & (pct_type3 < 20)] = 1  # "Good"

     return global_type

Ed_physic = pd.read_csv("/Users/charlotte.begouen/Downloads/detailed_5QC_outputs/2903787_QC.csv")
wavelengths = pd.to_numeric(Ed_physic.columns[10:-5])

# For 3, 4 and 5 wavelengths, we use fixed base wavelengths based on the existing QC methods
base_wavelengths = [380, 440, 490]
base_4 = base_wavelengths + [555]
base_5 = base_4 + [620]


# 1. Get list of unique profile IDs
all_profiles = Ed_physic["CYCLE"].unique()

# 2. Choose 10 random profiles
random_profiles = random.sample(list(all_profiles), 30)

# 3. Store results
all_quality_results = {}

# 4. Loop through each selected profile
for current_cycle in random_profiles:
    # Select your profile
    Ed_profile = Ed_physic[Ed_physic["CYCLE"] == current_cycle]
    Ed_profile = Ed_profile.reset_index(drop=True)

    all_quality_results[current_cycle] = {}

    # We are selecting 18 different numbers of wavelengths and will run the QC for each.
    # We always fix the original 5 wavelengths.
    for n_wl in [3, 4, 5, 6] + list(range(9, len(wavelengths) + 1, 4)):
        if n_wl == 3:
            qc_wls = base_wavelengths
        elif n_wl == 4:
            qc_wls = base_4
        elif n_wl == 5:
            qc_wls = base_5
        else:
            remaining_wls = [wl for wl in wavelengths if wl not in base_5]
            random_wls = random.sample(remaining_wls, n_wl - 5)
            qc_wls = sorted(base_5 + random_wls)

        final_wl = find_closest_wavelengths(qc_wls, wavelengths)

        # Filter Ed_profile to include only the selected wavelengths
        selected_columns = ['CYCLE', 'TIME', 'LON', 'LAT', 'PRES_FLOAT', 'Post_Pres', 'Post_Tilt'] + \
                           [f'{wv}' for wv in final_wl]
        Ed_profile_filtered = Ed_profile[selected_columns]

        # Apply the QC_all_wavelengths function
        tab_wmo_qualified = QC_all_wavelengths(
            tab_wmo=Ed_profile_filtered,
            wmo=current_cycle,
            sensor='Ed',
            nonshaded_min=0,
            to_save='/Users/charlotte.begouen/PycharmProjects/Hyper_Argo',
            columns_to_skip=7
        )

        # Compute the Global type
        tab_wmo_qualified['global_flag'] = compute_global_type(tab_wmo_qualified)
        cycle_global = tab_wmo_qualified.groupby('CYCLE')['global_flag'].first().reset_index()

        all_quality_results[current_cycle][n_wl] = cycle_global['global_flag'].iloc[0]

# All quality results will have a row for each current cycle, and a column for each n_wavelengths tested. It will be filled with the quality result.
df_all_quality_results = pd.DataFrame.from_dict(all_quality_results, orient='index')


# Reshape df_all_quality_results to long format for easier processing
df_long = df_all_quality_results.reset_index().melt(
    id_vars='index',
    var_name='n_wavelengths',
    value_name='quality'
).rename(columns={'index': 'current_cycle'})

# Create a pivot table: rows = n_wavelengths, columns = quality, values = counts
quality_counts = df_long.pivot_table(
    index='n_wavelengths',
    columns='quality',
    aggfunc='size',
    fill_value=0
)

# Sort by number of wavelengths on x-axis
plt.ylim(0, 30)

qc = quality_counts = quality_counts.sort_index()

x = qc.index.to_numpy()

bottom = np.zeros_like(x, dtype=int)
colors = ['#28a745', '#ffaa33', '#d62728']  # Vibrant green, lighter orange, and red

plt.figure(figsize=(8, 5))
lgd = ['Good', 'Questionable', 'Bad']
bar_width = 0.9  # Adjust bar width

plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, col in enumerate(qc.columns):
    plt.bar(x, qc[col], bottom=bottom, label=lgd[i], color=colors[i], width=bar_width)
    bottom += qc[col].to_numpy()

# Set x-axis ticks with fewer labels for better readability
tick_step = 5  # Adjust this value to control the spacing between ticks
plt.xticks(ticks=np.arange(min(x), max(x) + 1, tick_step), fontsize=12)

# Plot settings remain the same
plt.xlabel('Number of Wavelengths Used', fontsize=14)
plt.ylabel('Number of Profiles', fontsize=14)
plt.ylim(0, 30)  # Set y-axis range

plt.tight_layout()

# Save and display the plot
plt.savefig("quality_stacked_plot_nicer.png", dpi=300)
plt.show()