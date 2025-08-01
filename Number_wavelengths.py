import random

import pandas as pd

all_wavelengths = wavelengths

base_wavelengths = [380, 440, 490]
base_4 = base_wavelengths + [555]
base_5 = base_4 + [620]

# To store results
quality_results = []

# 1. Get list of unique profile IDs
all_profiles = Ed_physic["profile"].unique()

# 2. Choose 10 random profiles
random_profiles = random.sample(list(all_profiles), 30)

# 3. Store results
all_quality_results = []

# 4. Loop through each selected profile
for current_cycle in random_profiles:
    # Select your profile
    Ed_profile = Ed_physic[Ed_physic["profile"] == current_cycle]
    Ed_profile = Ed_profile.rename(
        columns=lambda col: float(col[2:]) if col.startswith("ed") and col[2:].replace('.', '', 1).isdigit() else col)
    wavelengths = [col for col in Ed_profile.columns if isinstance(col, (int, float))]
    Ed_profile = Ed_profile.reset_index(drop=True)

    for n_wl in [3, 4, 5, 6] + list(range(9, len(all_wavelengths) + 1, 4)):
        if n_wl == 3:
            qc_wls = base_wavelengths
        elif n_wl == 4:
            qc_wls = base_4
        elif n_wl == 5:
            qc_wls = base_5
        else:
            remaining_wls = [wl for wl in all_wavelengths if wl not in base_5]
            random_wls = random.sample(remaining_wls, n_wl - 5)
            qc_wls = sorted(base_5 + random_wls)
        final_wl = find_closest_wavelengths(qc_wls, wavelengths)


        results = Organelli_QC_Shapiro.organelli16_qc(
            Ed_profile, lat=Ed_profile["lat"].iloc[0], lon=Ed_profile["lon"].iloc[0],
            qc_wls=final_wl,
            step2_r2=0.995, step3_r2=0.997, step3_r3=0.999,
            skip_meta_tests=False
        )

        df_flags = pd.DataFrame(index=range(len(Ed_profile.depth)))
        df_results = pd.DataFrame()
        for result in results:
            global_flag, flags, status, polynomial_fit, wv = result
            new_flags = np.full(len(df_flags), 2)
            new_flags[:len(flags)] = flags
            df_flags[wv] = new_flags
            new_row = pd.DataFrame({
                'global_flag': [global_flag],
                'status': [status],
                'polynomial_fit': [polynomial_fit],
                'wavelength': [wv]
            })
            df_results = pd.concat([df_results, new_row], ignore_index=True)

        df_results = df_results.dropna(how='all').reset_index(drop=True)

        df_results_filtered = df_results[df_results['wavelength'] < 660]
        count = len(df_results_filtered)

        if n_wl ==5 :

            # Count the occurrences of each global_flag
            flag_counts = df_results['global_flag'].value_counts()

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
            quality,message  = conditions.get((count_0, count_1, count_2))

        else:
            if count == 0:
                quality = np.nan
            elif ((df_results_filtered['global_flag'] == 2).sum() / count >= 0.8 or
                  (df_results_filtered['global_flag'] == 1).sum() / count == 1):
                quality = 2
            elif ((df_results_filtered['global_flag'] == 0).sum() / count >= 0.8):
                quality = 0
            else:
                quality = 1

        print(f"Profile {current_cycle}, {n_wl} wavelengths -> Quality flag: {quality}")

        quality_results.append({
            'profile': current_cycle,
            'n_wavelengths': n_wl,
            'quality': quality,
            'wavelengths_used': qc_wls
        })

    # Convert to DataFrame
df_quality_summary = pd.DataFrame(quality_results)
    # Save to csv
df_quality_summary.to_csv(f"quality_summary_diffWV.csv", index=False)
#
# df_quality_summary = pd.read_csv('quality_summary_diffWV.csv')


# Create a pivot table: rows = n_wavelengths, columns = quality, values = counts
quality_counts = df_quality_summary.pivot_table(
    index='n_wavelengths',
    columns='quality',
    aggfunc='size',
    fill_value=0
)

# Sort by number of wavelengths on x-axis
quality_counts = quality_counts.sort_index()

# Use your existing DataFrame
qc = quality_counts.sort_index()
x = qc.index.to_numpy()
bottom = np.zeros_like(x, dtype=int)
colors = ['#2ca02c','#ffcc80', '#ff9999']   # Quality 0, 1, 2

plt.figure(figsize=(8, 5))
lgd = ['Good', 'Questionable', 'Bad']

# Iterate over quality levels and stack bars
for i, col in enumerate(qc.columns):
    plt.bar(x, qc[col], bottom=bottom, label=lgd[i], color=colors[i])
    bottom += qc[col].to_numpy()

plt.xlabel('Number of Wavelengths Used')
plt.ylabel('Number of Profiles')
plt.title('Profiles per QC type by Number of Wavelengths used for QC')
plt.xticks(x)  # ensure all x ticks are shown
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("quality_stacked_plot.png", dpi=300)
plt.show()

#import numpy as np



#%%

# Extract columns that start with 'ed'
# Ensure column names are strings before filtering
ed_columns = [col for col in map(str, Ed_physic.columns) if col.startswith('ed')]
# Extract wavelengths from the column names and convert them to numbers
wavelengths = []
for col in ed_columns:
    match = re.search(r'ed(\d+\.?\d*)', col)  # Match 'ed' followed by numbers
    if match:
        wavelengths.append(float(match.group(1)))  # Convert to float
    else:
        wavelengths.append(np.nan)  # Handle cases where no match is found

wavelengths = np.array(wavelengths)  # Convert to a NumPy array%%
