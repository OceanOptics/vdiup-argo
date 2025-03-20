from datetime import timedelta
import logging

import cmocean
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate, interp1d
from scipy.optimize import curve_fit


def fit_klu(df, fit_method='standard', wl_interp_method='pchip', smooth_method='triangular', only_continuous_obs=True):
    """
    Compute diffuse attenuation coefficient (KL) and extrapolate Lu to surface
    :param df: DataFrame containing datetime, depth, bin_counts, and Lu(lambda)
    :param fit_method: Method to fit Kl:
        + Standard: ordinary least-squares fit (very fast)
        + Semi-Robust: two-step ordinary least-squares fit, fit, remove residual, and refit (fast)
        + Robust: robust fit using trust region reflective algorithm to perform minimization (slow)
        + Iterative: compute penetration depth (zpd) and fit data from the surface to penetration depth only
        iteration stops if zpd change by less than 0.1 m or 10 iterations are reached whichever comes first
    :param wl_interp_method: Fit occasionally fails at some wavelength, data from these wavelength is retrieved
        from near-by wavelength by interpolation, accepted methods are 'pchip', 'linear', or any method
        supported by scripy.interpolate.interp1d. To disable interpolation set to None
    :param smooth_method: smooth spectrum spectrally post fit (n=7)
        + median: compute median on centered window of size n
        + gaussian: compute Gaussian weighted mean on window of size n and sigma=3
        + triangular: compute triangular weighted mean on window of size n
        + None: no smoothing (disable method)
    :param only_continuous_obs: ignore data sampled at less than 0.05Hz (1 sample/ 20 seconds)
    :return: DataFrame containing fitted Lu (Luf) and diffuse attenuation coefficient (Kl)
    """
    import logging

    no_data_above_zpd = False  # Initialize the flag

    logger = logging.getLogger('fit_klu')
    MIN_N_TO_FIRST_FIT = 10
    MIN_N_TO_ITER_FIT = 5
    # DETECT_LIMIT = 0.01  # Currently not computing but could be done in the future.
    df = df.sort_values(by='depth', ascending=True)

    # Init output
    wavelengths = [float(col[2:]) for col in df.columns if col.startswith(('ed', 'LU'))]
    if np.isnan(wavelengths).any():
        raise ValueError("wavelengths array contains NaN values")
    # Initialize output
    result = pd.DataFrame(np.full((len(wavelengths), 5), np.nan), columns=['Luf', 'Luf_sd', 'Kl', 'data_count','Kd_sd'],)

    # Remove values where z is above water
    mask = df['depth'] >= 0 | np.isnan(df['depth'])
    df = df[mask]
    # Check input once
    if len(df) < MIN_N_TO_FIRST_FIT:
        logger.warning('Not enough points to fit Kl')
        return result
    # Select columns to use
    col_sel = ~(df.filter(like='ed').isna().all(axis=0) | df.filter(like='LU').isna().all(axis=0))
    wk_sel = col_sel
    if 1 > sum(col_sel):
        logger.warning('Not enough channels to fit Kl')
        return result

    result = result.reindex(col_sel.index)
    idx_end = len(df) - 1

    if only_continuous_obs:  # Only high frequency sampling
        idx_end = (df['datetime'].diff(-1).abs() > pd.Timedelta(seconds=20)).idxmax()

    # Select rows based on the condition that none of the values in the selected "LU" columns are NaN
    lu_columns = [col for col in df.columns if col.startswith(('LU', 'ed'))]

    # Define helper function
    if fit_method == 'robust':
        def robust_polyfit(x, y):
            try:
                popt, pcov = curve_fit(lambda w, c0, c1: c0 * w + c1,
                                       x.to_numpy(), y.to_numpy(), method='trf', loss='soft_l1', f_scale=1.0)
            except RuntimeError:
                popt = np.array([np.nan, np.nan])
            return popt
    # Get data to use for fit
    # tic = time()
    row_sel = ~(df.loc[:idx_end, lu_columns].isnull().all(axis=1))
    z = df.loc[:idx_end, 'depth'][row_sel]
    lu = df.loc[:idx_end, lu_columns][row_sel]

    # Keep only data above detection limit
    lu = lu.loc[:, col_sel].astype(float)
    # Replace 0 by minimum value from instrument to prevent log to blow to infinity and replace negative value with 0
    lu[lu <= 0] = 1e-6
    # Replace infinite values with NaN
    lu.replace([np.inf, -np.inf], np.nan, inplace=True)


    if fit_method == 'robust':  # Robust exponential fit (slow, reference time)
        # Use exp
        # foo = lu.apply(lambda y: robust_exp_fit(z, y), axis='index')  # Same speed as robust polynomial fit
        # result['Luf'] = foo.iloc[0, :].to_numpy(float)
        # result['Kl'] = foo.iloc[1, :].to_numpy(float)
        # Use log
        c = np.log(lu).apply(lambda y: robust_polyfit(z, y), axis='index')
        result.loc[wk_sel, 'Luf'] = np.exp(c.iloc[1, :]).to_numpy(float)
        result.loc[wk_sel, 'Kl'] = -c.iloc[0, :].to_numpy(float)
    elif fit_method == 'standard':  # Ordinary least squares polynomial fit (fast, x60)
        c = np.log(lu).apply(lambda y: np.polyfit(z, y, 1), axis='index')
        result.loc[wk_sel, 'Luf'] = np.exp(c.iloc[1, :]).to_numpy(float)
        result.loc[wk_sel, 'Kl'] = -c.iloc[0, :].to_numpy(float)
    elif fit_method == 'semi-robust':  # Semi-robust least-square polynomial fit (still fast x15))
        c = np.log(lu).apply(lambda y: np.polyfit(z, y, 1), axis='index')
        pred = c.apply(lambda cc: np.polyval(cc, z), axis='index')
        pred.index = z.index
        res = np.log(lu) - pred  # Compute residual
        s = np.abs(res) <= 2 * np.std(res, 0)  # Ignore residuals too far from original fit
        n = int(len(z) / 2)
        c = np.log(lu[s]).apply(lambda y: (np.polyfit(z[~np.isnan(y)], y[~np.isnan(y)], 1)
                                           if sum(~np.isnan(y)) > 10 else np.array([np.nan, np.nan])), axis='index')
        result.loc[wk_sel, 'Luf'] = np.exp(c.iloc[1, :]).to_numpy(float)
        result.loc[wk_sel, 'Kl'] = -c.iloc[0, :].to_numpy(float)

    elif fit_method == 'iterative':  # Iterative changing penetration depth to which fit data
        lu = lu.apply(pd.to_numeric, errors='coerce')
        lu_log = lu.apply(np.log)
        # Compute a first penetration depth using 2nd order polynomial as in Zing et al. 2020

        def guess_zpd(z, lu_log):
            # Filter out NaN values for the current column
            valid_mask = ~np.isnan(lu_log)
            z_valid = z[valid_mask]
            lu_log_valid = lu_log[valid_mask]

            if len(z_valid) < 10 or len(lu_log_valid) < 10:  # Ensure there are enough points to fit
                #print('Not enough points to fit zpd')
                zpd0 = np.nan
                return zpd0

            # Perform polynomial fit
            try:
                fit = np.polynomial.polynomial.Polynomial.fit(z_valid, lu_log_valid, 2)
                c = fit.convert().coef
            except np.linalg.LinAlgError:
                # Handle the case where the polynomial fitting fails
                print("Polynomial fitting failed due to SVD not converging.")
                zpd0 = np.nan
                return zpd0

            if len(c) < 3:
                print("The polynomial fit did not return the expected number of coefficients.")
                zpd0 = np.nan
            else:
                # Use quadratic formula to find depth at which Lu(z) = Lu(0)/e
                b24ac = c[1] ** 2 - 4 * c[2]
                if b24ac >= 0:
                     zpd = np.array(((-c[1] + np.sqrt(b24ac)) / (2 * c[2]),
                                     (-c[1] - np.sqrt(b24ac)) / (2 * c[2])))
                     zpd0 = np.min(zpd) if np.all(zpd > 0) else np.max(zpd)
                     if zpd0 < 0:
                        zpd0 = np.nan
                else:  # Complex depth... default back to nan
                    zpd0 = np.nan
            return zpd0

        # Initialize an array to store zpd values for each wavelength
        zpd = np.full(lu_log.shape[1], np.nan)

        # Iterate over each column (wavelength) in lu_log
        for i, col in enumerate(lu_log.columns):
            zpd[i] = guess_zpd(z, lu_log[col])

        zpd = lu_log.apply(lambda y: guess_zpd(z, y), axis='index').to_numpy()
        # Iterate to find static zpd
        c = np.full((2, len(lu.columns)), np.nan)
        c_sd = np.empty_like(c)
        zpd_history = np.empty(10)
        for wli in range(c.shape[1]):
            valid_mask = ~np.isnan(lu_log.iloc[:, wli])
            z_valid = z[valid_mask]
            lu_log_valid = lu_log[valid_mask].iloc[:,wli]

            wk_sel_positions = [i for i, x in enumerate(wk_sel) if x]
            # Store the count of non-NaN data points
            result.iloc[wk_sel_positions[wli], result.columns.get_loc('data_count')] = len(lu_log_valid)

            if len(z_valid) < 10 or len(lu_log_valid) < 10 :
                continue

            zpd_history[1:] = float('nan')
            zpd_history[0] = zpd[wli]
            sel = z_valid < zpd[wli]
            if np.sum(sel) == 0:
                print('No data above zpd. Still calculating zpd/Kd')
                no_data_above_zpd = True  # Set the flag
                # zpd[wli] = np.nan
                # continue

            if np.sum(sel) < MIN_N_TO_ITER_FIT:
                # Start with at least 5 points (even if deeper than zpd)
                sel.iloc[:MIN_N_TO_ITER_FIT] = True
            for i in range(10):
                c[:, wli],cov = np.polyfit(z_valid[sel], lu_log_valid[sel], 1,cov=True)
                if c[0, wli] == 0:
                    zpd[wli] = np.nan  # Zpd cannot be computed.
                else:
                    zpd[wli] = -1 / c[0, wli]
                if zpd[wli] in zpd_history:
                    # Break if new zpd was already computed before (loop)
                    break
                if i > 5 and zpd[wli] > zpd_history[i - 1]:
                    # Break if after n/2 iteration the zpd keep increasing
                    #       main reason is that the iterative process is mostly used in the red where zpd is small
                    zpd[wli] = zpd_history[i - 1]  # reverse to smaller zpd
                    break
                seln = z_valid < zpd[wli]
                if (sel != seln).sum() <= 1 or seln.sum() < MIN_N_TO_ITER_FIT:
                    break
                sel = seln
                zpd_history[i] = zpd[wli]
            else:
                logger.debug(f'Fit max iteration reached (zpd={float(zpd[wli]):.1f}).')
            # Compute standard deviation
            c_sd[:,wli] = np.sqrt(np.diag(cov))
        result.loc[wk_sel, 'Luf'] = np.exp(c[1, :])
        result.loc[wk_sel, 'Kl'] = -c[0, :]

        result.loc[wk_sel, 'Luf_sd'] = c_sd[1, :] * result.loc[wk_sel, 'Luf']
        result.loc[wk_sel, 'Kd_sd'] = c_sd[0, :]
    else:
        raise ValueError(f'fit method: {fit_method} not supported.')
    # Interpolate missing wavelength
    #   Interpolate before smoothing as smoothing could shift data on large gaps
    #   No interpolation when below detection limit for 'Kl'

    for k in ('Luf', 'Kl'):
        sel = ~np.isfinite(result[k]) & wk_sel  # channels to interpolate
        nsel = np.isfinite(result[k]) & wk_sel  # channels to use
        if sum(sel) > 10 and k == 'Kl' and wl_interp_method !=  'None' :
            logger.warning('High number of fit failed')
        if sum(sel) == 0:  # No need to interpolate
            continue
        if wl_interp_method is None or smooth_method.lower() == 'none':
            pass
        elif wl_interp_method == 'pchip':
            result.loc[sel, k] = pchip_interpolate(wavelengths[nsel], result[k][nsel], wavelengths[sel])
        else:
            result.loc[sel, k] = interp1d(wavelengths[nsel], result[k][nsel], copy=False,
                                          assume_sorted=True,
                                          kind=wl_interp_method)(wavelengths[sel])

    # Smooth Spectra
    if smooth_method == 'median':
        result.Luf = result.Luf.rolling(3, center=True, min_periods=2).median()  # Median on square window
        result.Kl = result.Kl.rolling(7, center=True, min_periods=4).median()  # Median on square window
    elif smooth_method == 'gaussian':
        result.Luf = result.Luf.rolling(3, win_type='gaussian', center=True, min_periods=2).mean(std=1)
        result.Kl = result.Kl.rolling(7, win_type='gaussian', center=True, min_periods=4).mean(std=1)
    elif smooth_method == 'triangular':
        result.Luf = result.Luf.rolling(3, win_type='triang', center=True, min_periods=2).mean()
        result.Kl = result.Kl.rolling(7, win_type='triang', center=True, min_periods=4).mean()
    elif smooth_method is None or smooth_method.lower() == 'none':
        pass
    else:
        raise ValueError(f'smooth_method: {smooth_method} not supported.')


    # logger.debug(f"Kl fit in {time() - tic:.3f} seconds")
    return result, no_data_above_zpd
