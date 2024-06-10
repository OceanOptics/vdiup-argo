import numpy as np
from scipy.stats import norm
import pandas as pd
from sza_saa_grena import solar_zenith_and_azimuth_angle
import unittest
import pvlib
from statsmodels.stats.diagnostic import lilliefors
def organelli16_qc(df: object, lat: object = float('nan'), lon: object = float('nan'),
                   qc_wls: list = [490], step2_r2: object = 0.995, step3_r2: object = 0.998,
                   skip_meta_tests: object = False, skip_dark_test: object = True) -> object:
    """
    Quality control profile with Organelli et al. 2016
    Test is perfomed at the closest wavelength to `qc_wl`

    QC of location and sun_elevation is skipped if arguments sun_elevation, lat, and lon are None.

    :param time_utc: Time of surfacing of float in UTC time zone.
    :param df: Profile data frame, only take profile section (not surface data)
        requires fields DEPTH and `rad_key`
    :param sun_elevation: Sun elevation (degrees) = 90 - sun_zenith
    :param lat: Latitude (deg North)

    :param lon: Longitude (deg East)
    :param rad_key: Key to use in data frame for radiometric parameter to evaluate (e.g. LU, ES)
    :param qc_wl: Wavelength at which test is performed (nm)
    :param step2_r2: Squared Correlation coefficient for first test (default=0.995)
    :param step3_r2: Squared Correlation coefficient for second test (default=0.998)
    :param skip_meta_tests: Skip metadata test (boolean)
    :param skip_dark_test: Skip dark test (boolean)
    :return: (global_flag, local_flag, status,polynomial_fit)
        global_flag: True or False: passed or failed QC
        local_flag: flag for each observation
            0. good: does not require modification to be used
            2. questionable: potentially usable
            3. bad: need adjustment before use
        polynomial_fit: dictionary with polynomial fit information
        status: step at which failed QC
    """

    results = []


    for qc_wl in qc_wls:

        try:

            good, questionable, bad = 0, 1, 2
            flags = np.zeros(len(df), dtype=int) # Assume all good at beginning.

            # Extract wavelength to format for Kd function and rename columns
            wl = [col for col in df.columns if isinstance(col, (int, float))]
            wl_int = [int(w) for w in wl]
            wli = wl_int[np.argmin(abs(np.array(wl_int) - qc_wl))]
            rad = df[wli]

            # -- Step 0: Pre-checks + Sun Elevation --
            # Check enough observations
            if sum(~rad.isna().astype(bool), 0) < 10:
                results.append((False, flags, 'BAD: NO RES: TOO_FEW_OBS', False, qc_wl))
                continue
            if not skip_meta_tests:
                # Check monotonic profile
                DEPTH_array = df['DEPTH'].values
                if len(np.where(np.diff(DEPTH_array) <= 0)[0]) != 0:
                    # Flag non-monotonic values
                    flag_noMono = np.where(np.diff(DEPTH_array) <= 0)[0]
                    flags[flag_noMono] = 2  # Bad flag, not monotonous values

                    # Ignore non-monotonic rows for the rest of the code
                    rad[flag_noMono] = np.nan
                    df.loc[flag_noMono,'DEPTH'] = np.nan
                    print(f"Non-monotonic values detected at rows {flag_noMono}")

            # Check Location
            if not (-90 <= lat <= 90 and -180 <= lon <= 360):
                results.append(( False, flags, 'BAD: NO RES: INVALID_LOCATION', False,qc_wl))
                continue
                # Check Sun Elevation
                # sun_elevation = solar_zenith_and_azimuth_angle(lon, lat, time_utc.to_pydatetime())
                # if not 2 <= sun_elevation:
                #     return False, flags, 'NIGHTTIME'

            # # -- Step 1: Dark test --
            if not skip_dark_test:
                # Find dark section of profile
                i, converged = 0, True
                while lilliefors(pd.DataFrame(rad, dtype=float).iloc[i:])[1] < 0.01:
                    if len(df) - i < 10:
                        converged = False
                        break
                    i += 1  # Start deeper
                # Compute dark (not needed for HyperNav as integrated dark shutter is included)
                dark_wl = np.mean(rad.iloc[i:]) if converged else np.nan
                rad = rad - dark_wl

                # Check valid number of observations
                if i < 5:
                    results.append( (False, flags, 'TOO_FEW_OBS_DARK', False,qc_wl))
                    continue

            # -- Step 2: Cloud signal --
            # Check fit quality
            pd.options.mode.chained_assignment = None
            rad[rad <= 1e-6] = 1e-6 # Minimum value from the sensor
            sel = ~flags.astype(bool)
            log_rad = np.log(pd.to_numeric(rad[sel], errors='coerce'))

            # Remove rows with NaN values
            valid_indices = ~np.isnan(log_rad)
            DEPTH_valid = df.DEPTH[sel][valid_indices]
            log_rad_valid = log_rad[valid_indices]

            #  Fit polynomial
            p = np.polyfit(DEPTH_valid, log_rad_valid, 4)
            y = np.polyval(p, df.DEPTH[sel])

            mx = np.ma.masked_array(log_rad, mask=np.isnan(log_rad))
            my = np.ma.masked_array(y, mask=np.isnan(y))
            # Compute correlation coefficient
            r2 = np.ma.corrcoef(mx.flatten(), my)[0, 1] ** 2
            res = np.abs(np.array(log_rad).flatten() - y)
            valid_indices = ~np.isnan(res)
            # Flag individual observations (> 2 std)
            flags[np.argwhere(sel)[valid_indices][res[valid_indices] > 2 * np.std(res[valid_indices])]] = bad

            if r2 < step2_r2:
                results.append( (False, flags, 'BAD: CLOUDY_PROFILE', False, qc_wl))
                continue
             # Check valid number of observations
            if sum(~flags.astype(bool)) < 5:
                results.append(( False, flags, 'BAD: TOO_FEW_OBS_CLOUDY', False,qc_wl))
                continue

            # -- Step 3: Wave focusing --
            # Check fit quality on flagged data
            sel = ~flags.astype(bool)
            log_rad = np.log(pd.to_numeric(rad[sel], errors='coerce'))

            # Remove rows with NaN values
            valid_indices = ~np.isnan(log_rad)
            DEPTH_valid = df.DEPTH[sel][valid_indices]
            log_rad_valid = log_rad[valid_indices]

            p = np.polyfit(DEPTH_valid, log_rad_valid, 4)
            y = np.polyval(p, df.DEPTH[sel])

            mx = np.ma.masked_array(log_rad, mask=np.isnan(log_rad))
            my = np.ma.masked_array(y, mask=np.isnan(y))
            # Compute correlation coefficient
            r2 = np.ma.corrcoef(mx, my)[0, 1] ** 2

            polynomial_fit = {
                'coefficients': p,
                'depths': df.DEPTH[sel],
                'fitted_values': y}

            # Flag individual observations
            res = np.abs(log_rad - y)
            flags[np.argwhere(sel)[res > 1 * np.std(res)]] = 1
            flags[np.argwhere(sel)[res > 2 * np.std(res)]] = 2

            if r2 < step3_r2:
                results.append((False, flags, 'BAD: WAVE_FOCUSING', False,qc_wl))
                continue

            if sum(~flags.astype(bool)) < 5:
                results.append(( False, flags, 'BAD: TOO_FEW_OBS_NOTFLAGGED', False, qc_wl))
                continue

            results.append((True, flags, 'GOOD', polynomial_fit, qc_wl))
        except Exception as e:
            print(f"An error occurred while processing wavelength {qc_wl}: {e}")
            continue
    return results
