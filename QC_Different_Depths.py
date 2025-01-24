import numpy as np
from scipy.stats import norm
import pandas as pd
from sza_saa_grena import solar_zenith_and_azimuth_angle
import unittest
import pvlib
from statsmodels.stats.diagnostic import lilliefors


def organelli16_qc(df, lat=float('nan'), lon=float('nan'), qc_wls=[490], step2_r2=0.995, step3_r2=0.997, step3_r3=0.999,
                   skip_meta_tests=False, skip_dark_test=True, associated_depths=None):
    """
    Quality control profile with Organelli et al. 2016

    :param df: Profile data frame, only take profile section (not surface data)
        requires fields depth and `rad_key`
    :param lat: Latitude (deg North)
    :param lon: Longitude (deg East)
    :param step2_r2: Squared Correlation coefficient for first test (default=0.995)
    :param step3_r2: Squared Correlation coefficient for second test (default=0.998)
    :param skip_meta_tests: Skip metadata test (boolean)
    :param skip_dark_test: Skip dark test (boolean)
    :param associated_depths: Dictionary with wavelengths as keys and associated depths as values (default=None)
    :return: (global_flag, local_flag, status, polynomial_fit, wavelength)
        global_flag: 0,1,2: Overall profile QC for given wavelength
        local_flag: flag for each observation
            0. good: does not require modification to be used
            1. questionable: potentially usable
            2. bad: need adjustment before use
        polynomial_fit: dictionary with polynomial fit information
        status: step at which failed QC
        wavelength: wavelength at which QC was performed
    """

    results = []
    # Filter the DataFrame to include only the upper 250 meters following Organelli

    for qc_wl in qc_wls:
        try:
            if associated_depths and qc_wl in associated_depths:
                associated_depth = associated_depths[qc_wl]
                df_filtered = df[df['depth'] <= associated_depth]
            elif qc_wl > 600:
                closest_to_zero = df.loc[df['depth'].idxmin()]
                threshold_value = 0.01 * closest_to_zero[qc_wl]
                associated_depth = 2 * df[df[qc_wl] >= threshold_value]['depth'].max()
                df_filtered = df[df['depth'] <= associated_depth]
                step2_r2 = 0.95
            else:
                df_filtered = df[df['depth'] <= 150]

            good, questionable, bad = 0, 1, 2
            flags = np.zeros(len(df_filtered), dtype=int)  # Assume all good at beginning.

            # Extract wavelength to format for Kd function and rename columns
            wl = [col for col in df_filtered.columns if isinstance(col, (int, float))]
            wl_int = [int(w) for w in wl]
            wli = wl_int[np.argmin(abs(np.array(wl_int) - qc_wl))]
            rad = df_filtered[wli]

            # -- Step 0: Pre-checks + Sun Elevation --
            # Check enough observations
            if sum(~rad.isna().astype(bool), 0) < 10:
                flags[:] = bad
                results.append((2, flags, 'BAD: NO RES: TOO_FEW_OBS', False, qc_wl))
                continue
            if not skip_meta_tests:
                # Check monotonic profile
                depth_array = df_filtered['depth'].values
                if len(np.where(np.diff(depth_array) <= 0)[0]) != 0:
                    # Flag non-monotonic values
                    flag_noMono = np.where(np.diff(depth_array) <= 0)[0]
                    flags[flag_noMono] = bad  # Bad flag, not monotonous values

                    # Ignore non-monotonic rows for the rest of the code
                    rad[flag_noMono] = np.nan
                    df_filtered.loc[flag_noMono, 'depth'] = np.nan
                    print(f"Non-monotonic values detected at rows {flag_noMono}")

            # Check Location
            if not (-90 <= lat <= 90 and -180 <= lon <= 360):
                flags[:] = bad
                results.append((2, flags, 'BAD: NO RES: INVALID_LOCATION', False, qc_wl))
                continue

            # -- Step 1: Dark test --
            if not skip_dark_test:
                # Find dark section of profile
                i, converged = 0, True
                while lilliefors(pd.DataFrame(rad, dtype=float).iloc[i:])[1] < 0.01:
                    if len(df_filtered) - i < 10:
                        converged = False
                        break
                    i += 1  # Start deeper
                # Compute dark (not needed for HyperNav as integrated dark shutter is included)
                dark_wl = np.mean(rad.iloc[i:]) if converged else np.nan
                rad = rad - dark_wl

                # Check valid number of observations
                if i < 5:
                    flags[:] = bad
                    results.append((2, flags, 'TOO_FEW_OBS_DARK', False, qc_wl))
                    continue

            # -- Step 2: Cloud signal --
            # Check fit quality
            pd.options.mode.chained_assignment = None
            rad[rad <= 1e-6] = 1e-6  # Minimum value from the sensor
            sel = ~flags.astype(bool)
            log_rad = np.log(pd.to_numeric(rad[sel], errors='coerce'))

            # Remove rows with NaN values
            valid_indices = ~np.isnan(log_rad)
            depth_valid = df_filtered.depth[sel][valid_indices]
            log_rad_valid = log_rad[valid_indices]

            # Fit polynomial
            p = np.polyfit(depth_valid, log_rad_valid, 4)
            y = np.polyval(p, df_filtered.depth[sel])

            mx = np.ma.masked_array(log_rad_valid, mask=np.isnan(log_rad_valid))
            my = np.ma.masked_array(y, mask=np.isnan(y))
            # Compute correlation coefficient
            r2 = np.ma.corrcoef(mx.flatten(), my)[0, 1] ** 2
            res = np.abs(np.array(log_rad).flatten() - y)
            valid_indices = ~np.isnan(res)
            # Flag individual observations (> 2 std)
            flags[np.argwhere(sel)[valid_indices][res[valid_indices] > 2 * np.std(res[valid_indices])]] = bad

            if r2 < step2_r2:
                flags[:] = bad
                results.append((2, flags, 'BAD: CLOUDY_PROFILE', False, qc_wl))
                continue
            # Check valid number of observations
            if sum(~flags.astype(bool)) < 5:
                flags[:] = bad
                results.append((2, flags, 'BAD: TOO_FEW_OBS_CLOUDY', False, qc_wl))
                continue

            # -- Step 3: Wave focusing --
            # Check fit quality on flagged data
            sel = ~flags.astype(bool)
            log_rad = np.log(pd.to_numeric(rad[sel], errors='coerce'))

            # Remove rows with NaN values
            valid_indices = ~np.isnan(log_rad)
            depth_valid = df_filtered.depth[sel][valid_indices]
            log_rad_valid = log_rad[valid_indices]

            p = np.polyfit(depth_valid, log_rad_valid, 4)
            y = np.polyval(p, df_filtered.depth[sel])

            mx = np.ma.masked_array(log_rad, mask=np.isnan(log_rad))
            my = np.ma.masked_array(y, mask=np.isnan(y))
            # Compute correlation coefficient
            r2 = np.ma.corrcoef(mx, my)[0, 1] ** 2

            polynomial_fit = {
                'coefficients': p,
                'depths': df_filtered.depth[sel],
                'fitted_values': y}

            # Flag individual observations
            res = np.abs(log_rad - y)

            if r2 < step3_r2:
                flags[:] = bad
                results.append((2, flags, 'BAD: WAVE_FOCUSING', False, qc_wl))
                continue

            if sum(~flags.astype(bool)) < 5:
                flags[:] = bad
                results.append((2, flags, 'BAD: TOO_FEW_OBS_NOTFLAGGED', False, qc_wl))
                continue

            if step3_r2 < r2 < step3_r3:
                flags[flags == 0] = questionable
                flags[np.argwhere(sel)[res > 2 * np.std(res)]] = bad
                results.append((1, flags, 'QUESTIONABLE', polynomial_fit, qc_wl))
                continue

            flags[np.argwhere(sel)[res > 1 * np.std(res)]] = questionable
            flags[np.argwhere(sel)[res > 2 * np.std(res)]] = bad
            results.append((0, flags, 'GOOD', polynomial_fit, qc_wl))
        except Exception as e:
            print(f"An error occurred while processing wavelength {qc_wl}: {e}")
            continue
    return results