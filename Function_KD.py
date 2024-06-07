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

    :param df: DataFrame containing datetime, depth, and Lu(lambda)
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
    logger = logging.getLogger('fit_klu')
    MIN_N_TO_FIRST_FIT = 10
    MIN_N_TO_ITER_FIT = 5
    # DETECT_LIMIT = 0.01  # Currently not computing but could be done in the future.

    # Init output
    wavelengths = [float(col[2:]) for col in df.columns if col.startswith(('ED', 'LU'))]
    result = pd.DataFrame(np.empty((len(wavelengths), 3), dtype=float) * np.nan,
                          index=wavelengths, columns=['Luf', 'Luf_sd', 'Kl'])
    # Remove values where z is above water
    mask = df['depth'] >= 0
    df = df[mask]
    # Check input once
    if len(df) < MIN_N_TO_FIRST_FIT:
        logger.warning('Not enough points to fit Kl')
        return result
    # Select columns to use
    col_sel = ~(df.filter(like='ED').isna().all(axis=0) | df.filter(like='LU').isna().all(axis=0))
    if 1 > sum(col_sel):
        logger.warning('Not enough channels to fit Kl')
        return result

    result.index = result.index.infer_objects()

    # Select rows to use
    idx_end = len(df) - 1
    if only_continuous_obs:  # Only high frequency sampling
        idx_end = (df['datetime'].diff(-1).abs() > pd.Timedelta(seconds=20)).idxmax()
    # Select rows based on the condition that none of the values in the selected "LU" columns are NaN
    lu_columns = [col for col in df.columns if col.startswith(('LU', 'ED'))]
    # Select rows where all values in the LU columns are not NaN
    row_sel = df[lu_columns].notna().all(axis=1)

    # Check input twice
    if sum(row_sel) < 10:
        logger.warning('Not enough points to fit Kl')
        return result
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
    row_sel = ~(df.loc[:idx_end, lu_columns].isnull().any(axis=1))
    z = df.loc[:idx_end, 'depth'][row_sel]
    lu = df.loc[:idx_end, lu_columns][row_sel]

    # Keep only data above detection limit
    #lu_surf = (lu.loc[:, col_sel].iloc[:3].mean()  # Mean surface lu
    #           .rolling(7, win_type='triang', center=True, min_periods=3).mean())  # Smooth spectra
    # adl_sel = lu_surf > DETECT_LIMIT
    # result.loc[col_sel & ~adl_sel, 'Luf'] = lu.loc[:, col_sel & ~adl_sel].iloc[:3].mean()
    # Get Lu for values below detection limit
    # Kd tend to 0 but is set to NaN and must be handled properly later on
    # adl_sel.index = col_sel.index
    wk_sel = col_sel  # f& adl_sel
    lu = lu.loc[:, wk_sel].astype(float)  # Makes copy to 64bit (typically in 32bit) Require copy as edit data in
    # next step...

    wk_sel.index = result.index
    # Replace 0 by minimum value from instrument to prevent log to blow to infinity and replace negative value with 0
    lu[lu <= 0] = 1e-6
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', 'divide by zero encountered in log')
    #     warnings.filterwarnings('ignore', 'overflow encountered in square')
    # Fit

    # Replace infinite values with NaN
    lu.replace([np.inf, -np.inf], np.nan, inplace=True)
    lu = lu.dropna()

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
        # TODO Use previous wavelength zpd for first guest (faster and likely smoother result)
        def guess_zpd(z, lu_log):
            fit = np.polynomial.polynomial.Polynomial.fit(z, lu_log, 2)
            #  fit = lu_log.apply(lambda y: np.polyfit(z, y, 2), axis='index')
            c = c = fit.convert().coef
            # f = np.exp(c[2] * z ** 2 + c[1] * z + c[0])  # Compute fit
            # Use quadratic formula to find depth at which Lu(z) = Lu(0)/e
            b24ac = c[1] ** 2 - 4 * c[2]
            if b24ac >= 0:
                zpd = np.array(((-c[1] + np.sqrt(b24ac)) / (2 * c[2]),
                                (-c[1] - np.sqrt(b24ac)) / (2 * c[2])))
                zpd0 = np.min(zpd) if np.all(zpd > 0) else np.max(zpd)
                if zpd0 < 0:
                    zpd0 = 10
            else:  # Complex depth... default back to 10 meters
                zpd0 = 10
            return zpd0

        zpd = lu_log.apply(lambda y: guess_zpd(z, y), axis='index').to_numpy()
        # Iterate to find static zpd
        c = np.empty((2, len(lu.columns)), dtype=float)
        lu_log = lu_log.to_numpy()  # switch to numpy as can't use apply (different iter by wl)
        zpd_history = np.empty(10)
        for wli in range(c.shape[1]):
            zpd_history[1:] = float('nan')
            zpd_history[0] = zpd[wli]
            sel = z < zpd[wli]
            if np.sum(sel) < MIN_N_TO_ITER_FIT:
                # Start with at least 5 points (even if deeper than zpd)
                sel.iloc[:MIN_N_TO_ITER_FIT] = True
            for i in range(10):
                c[:, wli] = np.polyfit(z[sel], lu_log[sel, wli], 1)
                zpd[wli] = -1 / c[0, wli]  # same as 1 / Ku
                if zpd[wli] in zpd_history:
                    # Break if new zpd was already computed before (loop)
                    break
                if i > 5 and zpd[wli] > zpd_history[i - 1]:
                    # Break if after n/2 iteration the zpd keep increasing
                    #       main reason is that the iterative process is mostly used in the red where zpd is small
                    #       and for HyperNav we only use upper 10 meters anyways to compute K
                    zpd[wli] = zpd_history[i - 1]  # reverse to smaller zpd
                    break
                seln = z < zpd[wli]
                if (sel != seln).sum() <= 1 or seln.sum() < MIN_N_TO_ITER_FIT:
                    break
                sel = seln
                zpd_history[i] = zpd[wli]
            else:
                logger.debug(f'Lu({lu.columns[wli]:.2f}): Fit max iteration reached (zpd={zpd[wli]:.1f}).')
        result.loc[wk_sel, 'Luf'] = np.exp(c[1, :])
        result.loc[wk_sel, 'Kl'] = -c[0, :]
    else:
        raise ValueError(f'fit method: {fit_method} not supported.')
    # Interpolate missing wavelength
    #   Interpolate before smoothing as smoothing could shift data on large gaps
    #   No interpolation when below detection limit for 'Kl'

    for k in ('Luf', 'Kl'):
        sel = ~np.isfinite(result[k]) & wk_sel  # channels to interpolate
        nsel = np.isfinite(result[k]) & wk_sel  # channels to use
        if sum(sel) > 10 and k == 'Kl':
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
    # Compute residual standard deviation
    lu_modeled = result.loc[wk_sel, 'Luf'].to_numpy() * np.exp(
        -np.outer(z, result.loc[wk_sel, 'Kl'].to_numpy()))
    t = np.std(lu - lu_modeled, axis=0)
    t.index = result.index
    result['Luf_sd'] = t

    # logger.debug(f"Kl fit in {time() - tic:.3f} seconds")
    return result
