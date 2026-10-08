"""
Compute hyperspectral Kd from BGC-Argo RAMSES floats.

This version reads the calibrated DOWN_IRRADIANCE_SPECTRUM directly from the
aux files on the GDAC (https://data-argo.ifremer.fr/aux/coriolis/{wmo}/)

Input variables read from each *_aux.nc file (on N_PROF where DOWN_IRRADIANCE_SPECTRUM
is listed in STATION_PARAMETERS):
    DOWN_IRRADIANCE_SPECTRUM          : (N_LEVELS, N_VALUES70) - Ed in scientific units (*)
    DOWN_IRRADIANCE_SPECTRUM_ADJUSTED : (N_LEVELS, N_VALUES70) - delayed-mode Ed
    DOWN_IRRADIANCE_SPECTRUM_WAVELENGTHS : (N_LEVELS, N_VALUES70) - wavelengths (nm)
    PRES, MTIME, JULD, LATITUDE, LONGITUDE

(*) The Argo netCDF unit attribute is W/m^2/nm, but the magnitudes match
    uW/cm^2/nm; PAR conversion below (* 1e-2) assumes uW/cm^2/nm. Verify
    against a known cycle.
"""

import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspecw
from matplotlib import gridspec

#%% FUNCTIONS AND PATHS
ROOT = '/Volumes/SD/VDIUP/Argo/CBD'
# ROOT = '/Users/nils/Data/VDIUP/Argo/CBD'
PROCESSED_PROFILES = os.path.join(ROOT, 'New_Outputs_NotRaw')

if not os.path.exists(PROCESSED_PROFILES):
    os.makedirs(PROCESSED_PROFILES)

sys.path.append(ROOT)
import seabass_maker as sb            # noqa: E402
import Function_KD                    # noqa: E402
import Organelli_QC_Shapiro           # noqa: E402
import warnings
warnings.filterwarnings('ignore', message='scipy.stats.shapiro: Input data has range zero')

# --- Config --------------------------------------------------------------
TARGET_QC_WAVELENGTHS = [380, 443, 490, 550, 620]  # 5-wv QC
PLOT_TARGET_WAVELENGTHS = [490, 555, 660]
WMO_LIST_FILE = os.path.join(ROOT, 'WMOvsNSerie.txt')
WATERCOEFF_FILE = 'watercoeff.csv'
BOOTSTRAP_RANDOM_SEED = 15

# Set to True to reprocess every cycle from scratch instead of resuming from
# saved {wmo}_Kd.csv. False = skip cycles already in the saved file.
FORCE_REPROCESS = False
# Diagnostic plotting: show Ed-vs-depth panels for each cycle as it's processed.
# Set to True for ad-hoc inspection, False (default) for production runs.
PLOT_ED_PROFILES_DIAGNOSTIC = True
PLOT_DIAGNOSTIC_WAVELENGTHS = [380, 443, 490, 555, 620]

# --- Burst-sampled-cycle handling ----------------------------------------
# Some floats fire the radiometer in bursts (multiple measurements at a fixed
# depth in rapid succession). When these bursts occur in the lit zone of the
# profile, the resulting per-depth scatter (often 5x at the surface from wave
# focusing) makes the polynomial fit in Organelli QC fail with R^2 << 0.995,
# so the cycle is rejected even though the data below the burst is fine.
#
# When enabled, process_cycle detects burst-sampled cycles and median-collapses
# all depth bins with multiple measurements before passing to QC. This is a
# no-op for normal floats (each depth has one measurement -> median == value).
#
# Detection: a cycle is treated as burst-sampled if any depth bin in the top
# BURST_SHALLOW_DEPTH_M meters has at least BURST_MIN_PER_BIN measurements.
# This catches deliberate surface bursts but ignores incidental duplicates
# deeper in the profile (e.g. floats that briefly hover near zdark).
BURST_DETECT_AND_BIN = True
BURST_BIN_DEPTH_RESOLUTION = 0.1   # meters; depth rounding for bin grouping
BURST_SHALLOW_DEPTH_M = 15.0       # only shallow duplicates count for detection
BURST_MIN_PER_BIN = 3              # >=3 measurements at one depth = burst

# =========================================================================
#%% Helpers
def find_closest_wavelengths(targets, available_wavelengths):
    """Return the available wavelengths closest to each requested target."""
    return [min(available_wavelengths, key=lambda x: abs(x - t)) for t in targets]
def extract_wavelengths_from_columns(df, pattern='ed'):
    """Extract numeric wavelengths embedded in column names like 'ed490.0'."""
    cols = [c for c in df.columns if c.startswith(pattern) and 'unc' not in c]
    out = []
    for c in cols:
        m = re.search(pattern + r'(\d+\.?\d*)', c)
        out.append(float(m.group(1)) if m else np.nan)
    return np.array(out)
def load_existing_outputs(out_dir, wmo):
    """
    Load previously-saved Kd / Ed0 / Ed CSVs for incremental processing.

    Returns
    -------
    kd, ed0, ed_phys : DataFrames (empty if file is missing or unreadable)
    processed_cycles : set[int] of cycle numbers already in the saved Kd file
    """
    paths = {
        'kd':  os.path.join(out_dir, f'{wmo}_Kd.csv'),
        'ed0': os.path.join(out_dir, f'{wmo}_Ed0.csv'),
        'ed':  os.path.join(out_dir, f'{wmo}_Ed.csv'),
    }
    def _read(p):
        if not os.path.exists(p):
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame()

    kd = _read(paths['kd'])
    ed0 = _read(paths['ed0'])
    ed_phys = _read(paths['ed'])

    processed = set()
    if not kd.empty and 'profile' in kd.columns:
        processed = set(pd.to_numeric(kd['profile'], errors='coerce')
                        .dropna().astype(int).tolist())
    return kd, ed0, ed_phys, processed
def find_ed_n_prof(data):
    """Return the N_PROF index that carries DOWN_IRRADIANCE_SPECTRUM, or None."""
    sp = data.STATION_PARAMETERS.values
    for n in range(sp.shape[0]):
        params = []
        for p in sp[n]:
            if isinstance(p, bytes):
                p = p.decode()
            elif not isinstance(p, str):
                continue
            params.append(p.strip())
        if 'DOWN_IRRADIANCE_SPECTRUM' in params:
            return n
    return None
def quick_plot_ed_profile(Ed_profile, wavelengths, wmo, current_cycle,
                          target_wavelengths=PLOT_DIAGNOSTIC_WAVELENGTHS):
    """Diagnostic: 5 single-wavelength panels + full spectrum heatmap vs depth."""
    depths = Ed_profile['depth'].values
    closest = [min(wavelengths, key=lambda x: abs(x - t)) for t in target_wavelengths]

    fig, axes = plt.subplots(1, len(closest) + 1, figsize=(3 * (len(closest) + 1), 6),
                             sharey=True)
    for ax, wv in zip(axes[:-1], closest):
        ed = Ed_profile[wv].values
        valid = ~np.isnan(ed) & (ed > 0)
        ax.scatter(ed[valid], depths[valid], s=8, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel(f'Ed @ {int(wv)} nm')
        ax.set_title(f'{int(wv)} nm')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('Depth (m)')

    # Full-spectrum heatmap on the right
    ed_block = Ed_profile[list(wavelengths)].values.astype(float)
    ed_block = np.where(ed_block > 0, ed_block, np.nan)
    ax = axes[-1]
    pcm = ax.pcolormesh(np.array(wavelengths), depths,
                        np.log10(ed_block), shading='auto', cmap='viridis')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title('log10(Ed) full spectrum')
    plt.colorbar(pcm, ax=ax, fraction=0.05, pad=0.02)

    axes[0].invert_yaxis()
    fig.suptitle(f'Float {wmo} cycle {current_cycle} — {len(Ed_profile)} levels', fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.close(fig)
def _decode(x, default=''):
    if isinstance(x, bytes):
        return x.decode().strip()
    if isinstance(x, str):
        return x.strip()
    return default
def load_wavelengths_from_meta(wmo, root):
    """Reconstruct DOWN_IRRADIANCE_SPECTRUM wavelengths from {wmo}_meta_aux.nc.
    Uses the PREDEPLOYMENT_CALIB_COEFFICIENT polynomial for DOWN_IRRADIANCE_SPECTRUM
    together with LAUNCH_CONFIG_PARAMETER pixel/binning parameters.
    Returns a 1-D float ndarray of wavelengths (rounded to nearest nm), or None.
    """
    import re
    meta_path = os.path.join(root, wmo, f'{wmo}_meta_aux.nc')
    if not os.path.exists(meta_path):
        return None
    try:
        ds = xr.open_dataset(meta_path)
        params = [_decode(p) for p in ds.PARAMETER.values]
        if 'DOWN_IRRADIANCE_SPECTRUM' not in params:
            ds.close(); return None
        i = params.index('DOWN_IRRADIANCE_SPECTRUM')
        coeff_str = _decode(ds.PREDEPLOYMENT_CALIB_COEFFICIENT.values[i])

        coeffs = {}
        for cname in ('c0s', 'c1s', 'c2s', 'c3s', 'c4s'):
            m = re.search(rf'{cname}\s*=\s*([\-0-9eE\.\+]+)', coeff_str)
            if m:
                coeffs[cname] = float(m.group(1))
        if len(coeffs) < 5:
            ds.close(); return None

        cfg_names = [_decode(n) for n in ds.LAUNCH_CONFIG_PARAMETER_NAME.values]
        cfg_vals  = np.asarray(ds.LAUNCH_CONFIG_PARAMETER_VALUE.values).flatten()
        def _cfg(name):
            for n, v in zip(cfg_names, cfg_vals):
                if n == name:
                    return int(float(v))
            return None
        imin = _cfg('CONFIG_RamsesAccOutputPixelBegin_NUMBER')
        imax = _cfg('CONFIG_RamsesAccOutputPixelEnd_NUMBER')
        nbin = _cfg('CONFIG_RamsesAccOutputBinningSize_NUMBER')
        ds.close()
        if None in (imin, imax, nbin) or nbin < 1:
            return None

        i_full  = np.arange(1, 256)
        wv_full = (coeffs['c0s']
                   + coeffs['c1s'] * (i_full + 1)
                   + coeffs['c2s'] * (i_full + 1) ** 2
                   + coeffs['c3s'] * (i_full + 1) ** 3
                   + coeffs['c4s'] * (i_full + 1) ** 4)
        wv_sel  = wv_full[imin - 1: imax]
        n_vals  = len(wv_sel) // nbin
        wv_binned = wv_sel[:n_vals * nbin].reshape(n_vals, nbin).mean(axis=1)
        result = np.round(wv_binned).astype(float)
        print(f"  Reconstructed {len(result)} wavelengths from {wmo}_meta_aux.nc "
              f"({result[0]:.0f}–{result[-1]:.0f} nm)")
        return result
    except Exception as e:
        print(f"  Meta-file wavelength reconstruction failed for {wmo}: {e}")
        return None
def _ylim_kwargs(values, lo, hi):
    """Return {'ylim': [...]} or {} when values are all NaN/non-finite."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {'ylim': [lo * v.min(), hi * v.max()]}


def read_ed_from_aux(filename, fallback_wavelengths=None):
    try:
        data = xr.open_dataset(filename)
    except (OSError, RuntimeError, ValueError) as e:
        # Corrupt/truncated/non-NetCDF file (often from an interrupted wget).
        print(f"  Cannot open {os.path.basename(filename)} ({type(e).__name__}: {e}); skipping")
        return None
    n_prof = find_ed_n_prof(data)
    if n_prof is None:
        return None

    # Pick adjusted vs. realtime based on PARAMETER_DATA_MODE for DOWN_IRRADIANCE_SPECTRUM.
    sp = [_decode(p) for p in data.STATION_PARAMETERS.values[n_prof]]
    pdm = [_decode(p, 'R') for p in data.PARAMETER_DATA_MODE.values[n_prof]]
    try:
        ed_idx = sp.index('DOWN_IRRADIANCE_SPECTRUM')
        ed_mode = pdm[ed_idx] if ed_idx < len(pdm) else 'R'
    except ValueError:
        ed_mode = 'R'

    if ed_mode in ('A', 'D'):
        ed_arr = data.DOWN_IRRADIANCE_SPECTRUM_ADJUSTED.sel(N_PROF=n_prof).values
        # Fall back to realtime if adjusted is empty (e.g. partially populated)
        if np.isnan(ed_arr).all():
            ed_arr = data.DOWN_IRRADIANCE_SPECTRUM.sel(N_PROF=n_prof).values
    else:
        ed_arr = data.DOWN_IRRADIANCE_SPECTRUM.sel(N_PROF=n_prof).values


    OLD_LONG_NAME = "Downwelling irradiance spectrum"
    long_name = data['DOWN_IRRADIANCE_SPECTRUM'].attrs.get('long_name', '')
    if long_name.strip() == OLD_LONG_NAME:
        ed_arr = ed_arr * 1e-2

    pres = data.PRES.sel(N_PROF=n_prof).values
    mtime_name = next((v for v in ('MTIME', 'DOWN_IRRADIANCE_SPECTRUM_MTIME')
                       if v in data.variables), None)
    if mtime_name is None:
        mtime_name = next((v for v in data.variables if str(v).endswith('MTIME')), None)
    if mtime_name is not None:
        mtime = data[mtime_name].sel(N_PROF=n_prof).values
    else:
        print(f"  No MTIME variable in {os.path.basename(filename)}; "
              "depth correction will be skipped")
        mtime = np.full(pres.shape, np.nan)

    wavelengths = None
    if 'DOWN_IRRADIANCE_SPECTRUM_WAVELENGTHS' in data.variables:
        wv_arr = data.DOWN_IRRADIANCE_SPECTRUM_WAVELENGTHS.sel(N_PROF=n_prof).values
        wv_valid = np.where(~np.isnan(wv_arr).all(axis=1))[0]
        if len(wv_valid) > 0:
            # Round to nearest integer (kept as float). The GDAC stores
            # wavelengths with high precision, but Organelli_QC_Shapiro expects
            # integer-keyed wavelengths internally. With ~6-7 nm channel
            # spacing all values stay unique.
            wavelengths = np.round(wv_arr[wv_valid[0]]).astype(float)

    if wavelengths is None and fallback_wavelengths is not None:
        wavelengths = np.asarray(fallback_wavelengths, dtype=float)
        # Sanity check: shape must match Ed array's wavelength axis
        if wavelengths.shape[0] != ed_arr.shape[1]:
            print(f"  Wavelength fallback length {wavelengths.shape[0]} != "
                  f"Ed shape {ed_arr.shape[1]} in {os.path.basename(filename)}; skipping")
            return None
        print(f"  Using cached wavelengths for {os.path.basename(filename)} "
              "(file missing DOWN_IRRADIANCE_SPECTRUM_WAVELENGTHS)")

    if wavelengths is None:
        print(f"  No wavelengths in {os.path.basename(filename)} and no fallback; skipping")
        return None

    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError(
            f"Rounding wavelengths produced duplicates in {os.path.basename(filename)}; "
            "channel spacing is too fine -- round to 1 decimal instead.")

    # Keep only levels with at least one non-NaN Ed value.
    keep = ~np.isnan(ed_arr).all(axis=1)
    ed_arr = ed_arr[keep]
    pres = pres[keep]
    mtime = mtime[keep]

    if ed_arr.shape[0] == 0:
        return None

    ed_df = pd.DataFrame(ed_arr, columns=list(wavelengths))

    juld = data.JULD.sel(N_PROF=n_prof).values
    lat = float(data.LATITUDE.sel(N_PROF=n_prof).values)
    lon = float(data.LONGITUDE.sel(N_PROF=n_prof).values)

    return {
        'ed_df': ed_df,
        'wavelengths': wavelengths,
        'pres': pres,
        'mtime': mtime,
        'juld': juld,
        'lat': lat,
        'lon': lon,
        'n_prof': n_prof,
        'data_mode': ed_mode,
    }
def bootstrap_fit_klu_depth(df, speed, wavelengths, n_iterations,
                            random_seed=BOOTSTRAP_RANDOM_SEED):
    """Bootstrap Kd and Ed0 by perturbing depth and dropping random samples."""
    bootstrap_kd = []
    bootstrap_ed0 = []
    rng = np.random.default_rng(random_seed)
    perturbations = rng.normal(loc=0, scale=1, size=n_iterations)

    for i, perturbation in enumerate(perturbations):
        df_resampled = df.copy()
        df_resampled['depth'] = df_resampled['depth'] + speed * perturbation

        for col in df.columns:
            if col.startswith('ed'):
                non_nan = df[col].dropna().index
                drop = pd.Series(non_nan).sample(frac=0.2, random_state=i)
                df_resampled.loc[drop, col] = np.nan

        try:
            result, _ = Function_KD.fit_klu(
                df_resampled, fit_method='iterative', wl_interp_method='None',
                smooth_method='None', only_continuous_obs=False, verbose=False)
            bootstrap_kd.append(result['Kl'].values)
            bootstrap_ed0.append(result['Luf'].values)
        except Exception as e:
            print(f"  bootstrap iter {i} failed: {e}")
            continue

    if not bootstrap_kd or np.isnan(bootstrap_kd).all():
        nans = pd.Series([np.nan] * len(wavelengths))
        return nans, nans, nans, nans

    kd_df = pd.DataFrame(bootstrap_kd, columns=result.index)
    ed0_df = pd.DataFrame(bootstrap_ed0, columns=result.index)
    return kd_df.median(), kd_df.std(), ed0_df.median(), ed0_df.std()
def plot_ed_profiles(df, wmo, kd_df, wv_target, wv_og, ed0, flags_df, depth_col='depth'):
    """Per-cycle figures: Ed(0-) spectrum, Kd spectrum, and three Ed-vs-depth panels."""
    kd_df = kd_df.replace('-9999', np.nan)
    ed0 = ed0.replace('-9999', np.nan)
    df = df.dropna(subset=[depth_col])

    ed_columns = [c for c in df.columns if c.startswith('ed')]
    kd_columns = [c for c in kd_df.columns
                  if c.startswith('kd') and not any(s in c for s in ('unc', '_se', '_bincount'))]
    kd_unc_columns = [c for c in kd_df.columns if c.startswith('kd') and 'unc' in c]
    ed0_columns = [c for c in ed0.columns if c.startswith('ed0') and 'unc' not in c]
    ed0_unc_columns = [c for c in ed0.columns if c.startswith('ed') and 'unc' in c]
    quality_col = next(c for c in kd_df.columns if c.startswith('quality'))
    ed_wavelengths = np.array(wv_og)

    closest_columns, closest_kd_columns, closest_0_columns, closest_indexs = [], [], [], []
    for wavelength in wv_target:
        if np.isnan(ed_wavelengths).all():
            closest_columns.append(np.nan)
            continue
        idx = int(np.nanargmin(np.abs(ed_wavelengths - wavelength)))
        closest_indexs.append(idx)
        closest_columns.append(ed_columns[idx])
        closest_kd_columns.append(kd_columns[idx])
        closest_0_columns.append(ed0_columns[idx])

    for cycle in kd_df['profile'].unique():
        if cycle not in pd.to_numeric(df['profile']).values:
            continue
        try:
            profile_number = str(df[pd.to_numeric(df['profile']) == cycle]['profile'].iloc[0]).zfill(3)
        except IndexError:
            print(f"Could not find profile number for cycle {cycle}")
            continue

        figure_path = os.path.join(PROCESSED_PROFILES, wmo, f"{wmo}_{profile_number}_fig.png")
        if os.path.exists(figure_path):
            print(f"Figure already exists at {figure_path}")
            continue
        if kd_df[kd_df['profile'] == cycle][kd_columns].isna().all().all():
            print(f"All values for cycle {cycle} are NaN. Skipping figure.")
            continue

        fig = plt.figure(figsize=(12, 12))
        flag = kd_df[kd_df['profile'] == cycle][quality_col].values[0]
        status = {0: 'PASSED', 1: 'QUESTIONABLE', 2: 'FAILED'}.get(int(flag) if not pd.isna(flag) else -1, '?')
        fig.suptitle(f'Float {wmo} Cycle {cycle}: QC {status}', fontsize=20)

        gs = gridspec.GridSpec(3, 3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        ax5 = fig.add_subplot(gs[2, 2])

        # Background: all other passing cycles in light grey
        for inner_cycle in kd_df['profile'].unique():
            if inner_cycle != cycle and kd_df[kd_df['profile'] == inner_cycle][quality_col].values[0] == 0:
                ax1.plot(wv_og, ed0[ed0['profile'] == inner_cycle][ed0_columns].values[0], color='lightgrey')
                ax2.plot(wv_og, kd_df[kd_df['profile'] == inner_cycle][kd_columns].values[0], color='lightgrey')

        kd_values = pd.to_numeric(kd_df[kd_df['profile'] == cycle][kd_columns].values[0])
        kd_unc_values = pd.to_numeric(
            kd_df[kd_df['profile'] == cycle][kd_unc_columns].values[0], errors='coerce')
        ed0_values = ed0[ed0['profile'] == cycle][ed0_columns].values[0]
        ed0_unc_values = ed0[ed0['profile'] == cycle][ed0_unc_columns].values[0]

        ax1.plot(wv_og, ed0_values, color='blue', linewidth=2)
        ax1.fill_between(wv_og, ed0_values - ed0_unc_values, ed0_values + ed0_unc_values,
                         color='blue', alpha=0.2)
        ax1.set(xlabel='Wavelength (nm)', ylabel='Ed(0-) Values',
                title='Hyperspectral Ed(0-)',
                xlim=[min(wv_og), 700],
                **_ylim_kwargs(ed0_values, 0.90, 1.20))

        ax2.plot(wv_og, kd_values, color='blue', linewidth=2)
        ax2.fill_between(wv_og, kd_values - kd_unc_values, kd_values + kd_unc_values,
                         color='blue', alpha=0.2)
        ax2.set(xlabel='Wavelength (nm)', ylabel='Kd Values',
                title='Hyperspectral Kd',
                xlim=[min(wv_og), 700],
                **_ylim_kwargs(kd_values, 0.90, 1.05))

        colors = ['blue', 'green', 'red']
        for idx, (ed_col, kd_col, ed0_col, ax) in enumerate(zip(
                closest_columns, closest_kd_columns, closest_0_columns, [ax3, ax4, ax5])):

            kd_value = kd_df[kd_df['profile'] == cycle][kd_col].values[0]
            kd_unc_value = kd_df[kd_df['profile'] == cycle][kd_col + '_unc'].values[0]
            ed0_s = ed0[ed0['profile'] == cycle][ed0_col].values[0]
            ed0_unc_value = ed0[ed0['profile'] == cycle][ed0_col + '_unc'].values[0]

            flags_prof = flags_df[pd.to_numeric(flags_df['profile']) == cycle]
            df_filtered = df[pd.to_numeric(df['profile']) == cycle]

            flags = flags_prof[f'flag_{ed_wavelengths[closest_indexs[idx]]}']\
                .reset_index(drop=True).reindex(df_filtered.index)
            good_flags = flags[flags == 0].index
            question_flags = flags[flags == 1].index

            ax.scatter(df_filtered[ed_col], df_filtered[depth_col],
                       label=f'{ed_col}', c=colors[idx], alpha=0.3, marker='x')
            ax.scatter(df_filtered[ed_col][question_flags], df_filtered[depth_col][question_flags],
                       c=colors[idx], alpha=0.3, marker='o')
            ax.scatter(df_filtered[ed_col][good_flags], df_filtered[depth_col][good_flags],
                       c=colors[idx], alpha=0.7, marker='o')

            new_depth = np.linspace(df_filtered[depth_col].min(), 50, len(df_filtered[depth_col]))
            ed_pred = ed0_s * np.exp(-kd_value * new_depth)
            ed_upper = (ed0_s - ed0_unc_value) * np.exp(-(kd_value + kd_unc_value) * new_depth)
            ed_lower = (ed0_s + ed0_unc_value) * np.exp(-(kd_value - kd_unc_value) * new_depth)
            ax.plot(ed_pred, new_depth, '--', color=colors[idx])
            ax.fill_betweenx(new_depth, ed_lower, ed_upper, color=colors[idx], alpha=0.2)
            ax.axhline(1 / kd_value, linestyle=':', color=colors[idx])
            ax.set(xlabel='ED Values', ylabel='Depth (m)', ylim=[0, 50],
                   title=f'{ed_col} nm')
            ax.invert_yaxis()

        plt.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)
def detect_and_median_bin_bursts(Ed_profile, bin_resolution=0.1,
                                  shallow_depth_m=10.0, min_per_bin=3):
    """
    Detect a burst-sampled cycle (multiple measurements at the same depth in
    the upper / fittable part of the profile) and median-collapse duplicate-
    depth rows to a single robust value.

    A cycle is treated as burst-sampled if any depth bin (rounded to
    `bin_resolution` meters) within the top `shallow_depth_m` meters has at
    least `min_per_bin` measurements. This catches deliberate surface bursts
    that wreck the Organelli polynomial fit, while ignoring incidental
    duplicates deeper in the profile (e.g. floats that briefly hover near
    zdark, where bursts don't enter the fit anyway).

    For burst cycles, ALL duplicate-depth bins (shallow and deep) are
    collapsed: numeric columns -> median, text columns (date, time) -> first.

    Returns (Ed_profile, was_binned, n_bins_collapsed).
    """
    depth = Ed_profile['depth'].values
    decimals = max(0, int(round(-np.log10(bin_resolution))))
    bins_all = np.round(depth, decimals)
    valid = ~np.isnan(bins_all)
    unique_bins, counts = np.unique(bins_all[valid], return_counts=True)

    shallow_mask = unique_bins < shallow_depth_m
    is_burst = bool(((shallow_mask) & (counts >= min_per_bin)).any())
    if not is_burst:
        return Ed_profile, False, 0

    n_collapsed = int((counts > 1).sum())

    # Median-bin by rounded depth; drop rows with NaN depth (defensive)
    df = Ed_profile.copy()
    df['_depth_bin'] = bins_all
    df = df.dropna(subset=['_depth_bin'])

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols and c != '_depth_bin']
    agg = {c: 'median' for c in numeric_cols}
    for c in text_cols:
        agg[c] = 'first'

    binned = (df.groupby('_depth_bin', as_index=False).agg(agg)
                .drop(columns=['_depth_bin'])
                .sort_values('depth')
                .reset_index(drop=True))
    # Preserve the original column order (groupby+agg can reorder)
    binned = binned[[c for c in Ed_profile.columns if c in binned.columns]]
    return binned, True, n_collapsed

def process_cycle(filename, base_filename, wmo, current_cycle, fallback_wavelengths=None):
    """Build the Ed dataframe + flags for one cycle. Returns dict or None."""
    if '001D' in filename:
        print('Dark file, skipping')
        return None

    ed_data = read_ed_from_aux(filename, fallback_wavelengths=fallback_wavelengths)
    if ed_data is None:
        print(f"  No Ed data in {os.path.basename(filename)}, skipping")
        return None

    ed_df = ed_data['ed_df']
    wavelengths = ed_data['wavelengths']
    pres_ed = ed_data['pres']
    mtime = ed_data['mtime']
    juld = ed_data['juld']
    lat = ed_data['lat']
    lon = ed_data['lon']

    if ed_df.shape[0] == 1:
        print('Ed profile only has 1 depth, skipping')
        return None

    # --- Time stamps + depth correction (was: speed * 2s offset) ---
    n = ed_df.shape[0]
    mt = mtime[:n]
    if not np.issubdtype(mt.dtype, np.timedelta64):
        # MTIME is float days relative to JULD; NaN becomes NaT
        mt = pd.to_timedelta(mt, unit='D', errors='coerce').to_numpy()
    DT = pd.to_datetime(np.array([juld] * n) - mt)
    if DT.isna().any():
        DT = pd.to_datetime(np.array([juld] * n))

    speed = np.full(n, np.nan)
    for i in range(1, n):
        dt = (DT[i] - DT[i - 1]).total_seconds()
        dpres = pres_ed[i] - pres_ed[i - 1]
        if dt > 0:
            speed[i] = dpres / dt
    delta_depth = np.where(np.isnan(speed), 0.0, speed * 2)
    depth = pres_ed - delta_depth
    depth[0] = pres_ed[0] - delta_depth[1] if n > 1 else pres_ed[0]

    # --- Metadata: TEMP/PSAL from the standard B-file ---
    metadata_ed = pd.DataFrame({
        'wt': np.nan, 'sal': np.nan,
        'lon': lon, 'lat': lat,
        'date': DT.strftime('%Y%m%d'),
        'time': DT.strftime('%H:%M:%S'),
        'depth': depth,
        '_speed': speed,  # <-- new line
    })
    try:
        data_base = xr.open_dataset(base_filename)
        bp = data_base.PRES.sel(N_PROF=0).values[:n]
        bt = data_base.TEMP.sel(N_PROF=0).values[:n]
        bs = data_base.PSAL.sel(N_PROF=0).values[:n]
        metadata_ed['wt'] = np.interp(metadata_ed.depth, bp, bt, left=np.nan, right=np.nan)
        metadata_ed['sal'] = np.interp(metadata_ed.depth, bp, bs, left=np.nan, right=np.nan)
    except FileNotFoundError:
        print(f"  No core B-file ({os.path.basename(base_filename)}); TEMP/PSAL left as NaN")

    Ed_profile = pd.concat([metadata_ed, ed_df], axis=1)

    # Skip if everything is NaN/inf in the spectra columns
    cols_to_check = wavelengths.tolist()
    spectra_block = Ed_profile[cols_to_check]
    if spectra_block.map(lambda x: pd.isna(x) or np.isinf(x)).all().all():
        print('All spectra values are NaN/inf, skipping')
        return None

    # --- Burst-sampled cycle handling ---
    # If this cycle has a burst-style sampling pattern (multiple measurements
    # at the same shallow depth), median-collapse duplicate-depth rows so the
    # downstream polynomial fit isn't dominated by per-burst scatter.
    if BURST_DETECT_AND_BIN:
        Ed_profile, was_binned, n_collapsed = detect_and_median_bin_bursts(
            Ed_profile,
            bin_resolution=BURST_BIN_DEPTH_RESOLUTION,
            shallow_depth_m=BURST_SHALLOW_DEPTH_M,
            min_per_bin=BURST_MIN_PER_BIN,
        )
        if was_binned:
            print(f"  Burst sampling detected: median-binned {n_collapsed} duplicate-depth bin(s)")

        # These two lines go OUTSIDE the if-block so they always run:
    speed = Ed_profile['_speed'].values
    Ed_profile = Ed_profile.drop(columns=['_speed'])

    # --- Diagnostic plot ---
    if PLOT_ED_PROFILES_DIAGNOSTIC:
        quick_plot_ed_profile(Ed_profile, wavelengths, wmo, current_cycle)


    # --- PAR ---
    # Original code converted Ed_profile values from uW/cm^2/nm to W/m^2/nm via *1e-2.
    # The GDAC unit attribute says W/m^2/nm but the magnitudes matched uW/cm^2/nm
    irr_conv = Ed_profile[wavelengths]
    photon_factor = np.array(wavelengths) * 1e-9 / (2.998e8 * 6.62606957e-34)
    par_band = (np.array(wavelengths) >= 350) & (np.array(wavelengths) <= 700)
    Ed_profile['Epar'] = (np.trapezoid((np.array(irr_conv)[:, par_band] * photon_factor[par_band]))
                          / 6.02214129e23) * 1e6 / 1e4  # umol photons / cm2 / s

    # --- Organelli QC ---
    results = Organelli_QC_Shapiro.organelli16_qc(
        Ed_profile, lat=lat, lon=lon, qc_wls=list(wavelengths),
        step2_r2=0.995, step3_r2=0.997, step3_r3=0.999)

    df_flags = pd.DataFrame(columns=list(wavelengths), index=range(len(Ed_profile)))
    results_records = []

    for global_flag, flags, status, polynomial_fit, wv in results:
        if len(flags) < len(df_flags):
            new_flags = np.full(len(df_flags), 2)
            new_flags[:len(flags)] = flags
        else:
            new_flags = flags
        df_flags[wv] = new_flags
        results_records.append({
            'global_flag': global_flag, 'status': status,
            'polynomial_fit': polynomial_fit, 'wavelength': wv,
        })
    df_results = pd.DataFrame(results_records)

    data_dict_flags = {
        'depth': Ed_profile['depth'].values,
        'profile': [current_cycle] * len(Ed_profile),
    }
    for w in wavelengths:
        data_dict_flags[f'flag_{w}'] = df_flags[w].values

    df_results_filt = df_results[df_results['wavelength'] < 600]
    n_total = len(df_results_filt)
    n_bad = (df_results_filt['global_flag'] == 2).sum()
    n_quest = (df_results_filt['global_flag'] == 1).sum()
    if n_bad / n_total > 0.5:
        Ed_profile['quality'] = 2
        print(f"Cycle {current_cycle}: BAD (>50% wavelengths failed QC)")
    elif n_quest / n_total > 0.5 or n_bad / n_total > 0.05:
        Ed_profile['quality'] = 1
        print(f"Cycle {current_cycle}: QUESTIONABLE")
    else:
        Ed_profile['quality'] = 0
        print(f"Cycle {current_cycle}: PASSED")

    return {
        'Ed_profile': Ed_profile,
        'wavelengths': wavelengths,
        'flags_df': df_flags,
        'flags_dict': data_dict_flags,
        'speed': speed,
    }
def fit_kd_for_cycle(cycle_data, current_cycle):
    """Run Kd fit (single + bootstrap) for one cycle, return (kd_record, ed0_record)."""
    Ed_profile = cycle_data['Ed_profile']
    wavelengths = cycle_data['wavelengths']
    df_flags = cycle_data['flags_df']
    speed = cycle_data['speed']

    base_record = {
        'profile': int(current_cycle),
        'date': Ed_profile.date[0],
        'time': Ed_profile.time[0],
        'lon': round(Ed_profile.lon[0], 5),
        'lat': round(Ed_profile.lat[0], 5),
        'quality': Ed_profile['quality'][0],
    }

    # If BAD, emit a record with NaN spectra
    if Ed_profile.loc[0, 'quality'] == 2:
        kd_rec = dict(base_record)
        for w in wavelengths:
            kd_rec[f'kd{w}'] = np.nan
        for w in wavelengths:
            kd_rec[f'kd{w}_unc'] = np.nan
        for w in wavelengths:
            kd_rec[f'kd{w}_se'] = np.nan
        for w in wavelengths:
            kd_rec[f'kd{w}_bincount'] = np.nan
        return kd_rec, None

    # Patch wavelengths whose flags are all-bad with the closest-to-555 column
    closest_555 = min(wavelengths, key=lambda x: abs(x - 555))
    for col in df_flags.columns:
        if int(col) < 700 and df_flags[col].eq(2).all():
            df_flags[col] = df_flags[closest_555]

    # Rename wavelength columns to ed{wv}, build the new_Ed dataframe
    Ed_renamed = Ed_profile.copy()
    for w in wavelengths:
        Ed_renamed.rename(columns={w: f'ed{w}'}, inplace=True)
    ed_cols = [c for c in Ed_renamed.columns if c.startswith('ed') and not c.startswith('ed0')]
    new_Ed = Ed_renamed.loc[:, ['date', 'time', 'depth'] + ed_cols].copy()
    for w, col in zip(wavelengths, ed_cols):
        bad_idx = df_flags[w][df_flags[w] == 2].index
        new_Ed.loc[bad_idx, col] = np.nan

    # Bootstrap + single-fit
    median_kd, std_kd, median_ed0, std_ed0 = bootstrap_fit_klu_depth(
        new_Ed, speed, wavelengths, n_iterations=100)
    result, no_data_above_zpd = Function_KD.fit_klu(
        new_Ed, fit_method='iterative', wl_interp_method='None',
        smooth_method='None', only_continuous_obs=False)
    if no_data_above_zpd:
        Ed_profile.loc[0, 'quality'] = 1
        base_record['quality'] = 1

    result_kd = result['Kl'].mask(result['Kl'] < 0, np.nan) if not np.isnan(median_kd).all() else result['Kl']
    se_kd = result['Kd_sd'] / np.sqrt(result['data_count'])
    kd_unc = std_kd / np.sqrt(10) if not np.isnan(median_kd).all() else pd.Series([np.nan] * len(wavelengths))
    if np.isnan(median_kd).all():
        std_ed0 = pd.Series([np.nan] * len(wavelengths))

    kd_rec = dict(base_record)
    ed0_rec = dict(base_record)

    # Group columns by suffix: all kd, then all kd_unc, then all kd_se, then all kd_bincount
    n_wv = len(wavelengths)
    for i, w in enumerate(wavelengths):
        kd_rec[f'kd{w}'] = result_kd.iloc[i] if i < len(result_kd) else np.nan
    for i, w in enumerate(wavelengths):
        kd_rec[f'kd{w}_unc'] = float(kd_unc.iloc[i]) if i < len(kd_unc) else np.nan
    for i, w in enumerate(wavelengths):
        kd_rec[f'kd{w}_se'] = se_kd.iloc[i] if i < len(se_kd) else np.nan
    for i, w in enumerate(wavelengths):
        kd_rec[f'kd{w}_bincount'] = result['data_count'].iloc[i] if i < len(result['data_count']) else np.nan

    # Same grouping for Ed0: all ed0, then all ed0_unc
    for i, w in enumerate(wavelengths):
        ed0_rec[f'ed0{w}'] = median_ed0.iloc[i] if i < len(median_ed0) else np.nan
    for i, w in enumerate(wavelengths):
        ed0_rec[f'ed0{w}_unc'] = float(std_ed0.iloc[i]) if i < len(std_ed0) else np.nan

    return kd_rec, ed0_rec

#%% DOWNLOAD FROM GDAC
df_wmo = pd.read_table(WMO_LIST_FILE)
list_wmo = df_wmo['WMO'].unique()

for number in list_wmo:
    dir_path = os.path.join(ROOT, str(number))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, 'profiles_general'))
        print(f"Created profile directory: {dir_path}")

    # aux: contains both raw counts and calibrated DOWN_IRRADIANCE_SPECTRUM
    cmd_aux = (f"wget -r -np --wait 1 -nH -N --cut-dirs=3 -P {dir_path} "
               f"--reject 'index.html*' "
               f"https://data-argo.ifremer.fr/aux/coriolis/{number}/")
    # standard B-files: needed for TEMP/PSAL interpolated to Ed depths
    cmd_general = (f"wget -r -np --wait 1 -nH -N --cut-dirs=4 "
                   f"-P {os.path.join(dir_path, 'profiles_general')} "
                   f"--reject 'index.html*' --accept 'R*.nc' "
                   f"https://data-argo.ifremer.fr/dac/coriolis/{number}/profiles/")
    subprocess.run(cmd_general, shell=True)
    subprocess.run(cmd_aux, shell=True)

    if number ==3902759:
        cmd_aux = (f"wget -r -np --wait 1 -nH -N --cut-dirs=3 -P {dir_path} "
                   f"--reject 'index.html*' "
                   f"https://data-argo.ifremer.fr/aux/aoml/{number}/")
        # standard B-files: needed for TEMP/PSAL interpolated to Ed depths
        cmd_general = (f"wget -r -np --wait 1 -nH -N --cut-dirs=4 "
                       f"-P {os.path.join(dir_path, 'profiles_general')} "
                       f"--reject 'index.html*' --accept 'R*.nc' "
                       f"https://data-argo.ifremer.fr/dac/aoml/{number}/profiles/")
        subprocess.run(cmd_general, shell=True)
        subprocess.run(cmd_aux, shell=True)

#%% PROCESSING
if __name__ == '__main__':

  # Grab the Wmos from existing directories (in case some were added manually or by a previous run)
    wmos = sorted([item for item in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, item))
                   and re.match(r"^\d+$", item)])

    # Step 3 -- Process each float
    comments = [
        'These data were collected and made freely available by the International Argo Program and the national programs',
        'that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the',
        'Global Ocean Observing System https://doi.org/10.17882/42182.',
        'Link to BGC-Argo GDAC for raw float data: https://data-argo.ifremer.fr/aux/coriolis/.',
        'Hyperspectral Ed read directly from DOWN_IRRADIANCE_SPECTRUM in the Argo aux files.',
        'Quality Flag relates to the overall radiometric quality control based on Organelli et al., 2016 (DOI: 10.1175/JTECH-D-15-0193.1).',
        'Quality control is performed at each wavelength, see documentation for details.',
        'The overall "quality" flag per profile is recorded based on performance of all wavelengths below 600nm with following definition:',
        '    0. Good: >50% of wavelengths passed individual QC.',
        '    1. Questionable: >50% of wavelengths are questionable following individual QC or >5% of wavelengths flagged as Bad.',
        '    2. Bad: >50% of wavelengths are bad following individual QC.',
        'Uncertainties (_unc) are computed with a bootstrap technique and encompass uncertainty in fitting Kd to the profile and depth uncertainty. Details available in documentation.',
    ]

    for wmo in wmos:
        print(f"\n=== Float {wmo} ===")

        if not os.path.exists(os.path.join(ROOT, wmo, 'profiles')):
            print(f'No profiles for float {wmo}')
            continue

        out_dir = os.path.join(PROCESSED_PROFILES, wmo)
        os.makedirs(out_dir, exist_ok=True)

        kd_records, ed0_records = [], []
        flags_dataframes = []
        Ed_physic = pd.DataFrame()

        if FORCE_REPROCESS:
            existing_Kd = pd.DataFrame()
            existing_Ed0 = pd.DataFrame()
            existing_Ed_physic = pd.DataFrame()
            processed_cycles = set()
        else:
            existing_Kd, existing_Ed0, existing_Ed_physic, processed_cycles = (
                load_existing_outputs(out_dir, wmo))
            if processed_cycles:
                print(f"  Resuming: {len(processed_cycles)} cycle(s) already processed")

        metadata = {
            'investigators': 'Nils_Haentjens,Charlotte_Begouen_Demeaux,Robert_Frouin,Jing_Tan',
            'affiliations': 'University_of_Maine,University_of_Maine,Scripps_Institute_of_Oceanography,Scripps_Institute_of_Oceanography',
            'contact': 'nils.haentjens@maine.edu',
            'experiment': 'PVST_VDIUP',
            'cruise': f'Argo_{wmo}',
            'platform_id': wmo,
            'instrument_manufacturer': 'TriOS',
            'instrument_model': 'RAMSES',
            'documents': 'PVST_VDIUP_float_documentation_R2.pdf',
            'calibration_files': 'no_cal_files',  # was the cals/ AllCal.txt
            'data_type': 'drifter',
            'data_status': 'preliminary',
            'water_depth': 'NA',
            'measurement_depth': 'NA',
        }

        aux_files = sorted(glob.glob(os.path.join(ROOT, wmo, 'profiles', '*_aux.nc')))

        # Per-float wavelength cache. Wavelengths are fixed per instrument,
        # so once we read them from any cycle (or from a previously-saved Kd
        # file) we can reuse them when a sibling cycle is missing the
        # DOWN_IRRADIANCE_SPECTRUM_WAVELENGTHS variable.
        float_wv_cache = None
        if not existing_Kd.empty:
            cached = extract_wavelengths_from_columns(existing_Kd, pattern='kd')
            if len(cached) > 0 and not np.isnan(cached).all():
                float_wv_cache = cached

        if float_wv_cache is None:
            float_wv_cache = load_wavelengths_from_meta(wmo, ROOT)

        for filename in aux_files:
            base = os.path.basename(filename)
            # Skip files whose WMO in the name doesn't match the directory's WMO.
            # This catches stale / misplaced files that would otherwise be
            # silently attributed to the wrong float.
            wmo_in_name = re.match(r"[A-Z]*(\d+)_", base)
            if wmo_in_name and wmo_in_name.group(1) != wmo:
                print(f"  WARNING: {base} is from float {wmo_in_name.group(1)}, "
                      f"not {wmo}; skipping")
                continue

            m = re.search(r"_([0-9]+).*_aux\.nc$", filename)
            if not m:
                continue
            current_cycle = m.group(1)

            if int(current_cycle) in processed_cycles:
                continue

            base_filename = os.path.join(
                ROOT, wmo, 'profiles_general', re.split(r'_aux\.nc', base)[0] + '.nc')


            cycle_data = process_cycle(filename, base_filename, wmo, current_cycle,
                                       fallback_wavelengths=float_wv_cache)
            if cycle_data is None:
                continue

            # First successful cycle of this float seeds the cache for siblings.
            if float_wv_cache is None:
                float_wv_cache = cycle_data['wavelengths']

            kd_rec, ed0_rec = fit_kd_for_cycle(cycle_data, current_cycle)
            kd_records.append(kd_rec)
            if ed0_rec is not None:
                ed0_records.append(ed0_rec)

            flags_dataframes.append(pd.DataFrame(cycle_data['flags_dict']))

            ed_with_station = cycle_data['Ed_profile'].copy()
            # Rename wavelength cols to ed{wv} for consistency in saved CSV
            for w in cycle_data['wavelengths']:
                ed_with_station.rename(columns={w: f'ed{w}'}, inplace=True)
            ed_with_station['profile'] = current_cycle
            Ed_physic = pd.concat([Ed_physic, ed_with_station], ignore_index=True)

        if not kd_records:
            if not existing_Kd.empty:
                print(f"  No new cycles for float {wmo} (already up to date).")
            else:
                print(f"  No usable cycles for float {wmo}.")
            continue

        Kd_new = pd.DataFrame(kd_records)
        Ed0_new = pd.DataFrame(ed0_records) if ed0_records else pd.DataFrame()
        flags_df_combined = pd.concat(flags_dataframes, ignore_index=True) if flags_dataframes else pd.DataFrame()

        # Merge with existing data, sort by profile for clean output
        Kd = pd.concat([existing_Kd, Kd_new], ignore_index=True) if not existing_Kd.empty else Kd_new
        if not Ed0_new.empty:
            Ed0 = pd.concat([existing_Ed0, Ed0_new], ignore_index=True) if not existing_Ed0.empty else Ed0_new
        else:
            Ed0 = existing_Ed0
        Ed_physic = pd.concat([existing_Ed_physic, Ed_physic], ignore_index=True) \
            if not existing_Ed_physic.empty else Ed_physic

        if 'profile' in Kd.columns:
            Kd = Kd.sort_values('profile', kind='stable').reset_index(drop=True)
        if not Ed0.empty and 'profile' in Ed0.columns:
            Ed0 = Ed0.sort_values('profile', kind='stable').reset_index(drop=True)
        if not Ed_physic.empty and 'profile' in Ed_physic.columns:
            Ed_physic = Ed_physic.sort_values(['profile', 'depth'], kind='stable').reset_index(drop=True)

        # --- Mask Kd values smaller than pure-water aw for wv > 700 ---
        try:
            watercoeff = pd.read_csv(WATERCOEFF_FILE)
            for col in [c for c in Kd.columns if c.startswith('kd') and 'unc' not in c
                        and '_se' not in c and '_bincount' not in c]:
                m = re.search(r'kd(\d+)', col)
                if not m:
                    continue
                wavelength = int(m.group(1))
                if wavelength <= 700:
                    continue
                aw_lookup = watercoeff.loc[watercoeff['lambda'] == wavelength, 'aw']
                if aw_lookup.empty:
                    continue
                aw_value = aw_lookup.values[0]
                bad = Kd[col] < aw_value
                Kd.loc[bad, col] = np.nan
                if f'{col}_unc' in Kd.columns:
                    Kd.loc[bad, f'{col}_unc'] = np.nan
                if not Ed0.empty and f'ed0{wavelength}.0' in Ed0.columns:
                    Ed0.loc[bad, f'ed0{wavelength}.0'] = np.nan
                    Ed0.loc[bad, f'ed0{wavelength}.0_unc'] = np.nan
        except FileNotFoundError:
            print(f"watercoeff file not found ({WATERCOEFF_FILE}); skipping aw mask.")

        # --- Sync NaNs across companion columns ---
        # Wherever kd{wv} is NaN, force kd{wv}_unc, kd{wv}_se, kd{wv}_bincount to NaN.
        # Same for ed0{wv} -> ed0{wv}_unc. Catches any path where kd was masked
        # (aw mask, BAD cycle, fit failure) without the companions being synced.
        kd_base_cols = [c for c in Kd.columns
                        if c.startswith('kd')
                        and not any(c.endswith(s) for s in ('_unc', '_se', '_bincount'))]
        for col in kd_base_cols:
            nan_mask = Kd[col].isna()
            for suffix in ('_unc', '_se', '_bincount'):
                companion = f'{col}{suffix}'
                if companion in Kd.columns:
                    Kd.loc[nan_mask, companion] = np.nan

        if not Ed0.empty:
            ed0_base_cols = [c for c in Ed0.columns
                             if c.startswith('ed0') and not c.endswith('_unc')]
            for col in ed0_base_cols:
                nan_mask = Ed0[col].isna()
                companion = f'{col}_unc'
                if companion in Ed0.columns:
                    Ed0.loc[nan_mask, companion] = np.nan

        # --- Save outputs ---
        Kd.to_csv(os.path.join(out_dir, f'{wmo}_Kd.csv'), index=False)
        print(f'Kd file for float {wmo} written')
        Ed0.to_csv(os.path.join(out_dir, f'{wmo}_Ed0.csv'), index=False)
        print(f'Ed0 file for float {wmo} written')
        Ed_physic.to_csv(os.path.join(out_dir, f'{wmo}_Ed.csv'), index=False)
        print(f'Ed file for float {wmo} written')

        # Per-month SeaBASS files: only rewrite months that contain new cycles
        new_cycle_ints = {int(r['profile']) for r in kd_records}
        Kd['year_month'] = pd.to_datetime(Kd['date']).dt.strftime('%Y%m')
        months_to_write = set(
            Kd.loc[Kd['profile'].astype(int).isin(new_cycle_ints), 'year_month'].unique()
        )
        for ym, group in Kd.groupby('year_month'):
            if ym not in months_to_write:
                continue
            group = group.drop(columns=['year_month'])
            sb.format_to_seabass(
                group, metadata, f'PVST_VDIUP-Argo-Kd_{wmo}_{ym}_R2', out_dir,
                comments, missing_value_placeholder='-9999', delimiter='comma')
        Kd = Kd.drop(columns=['year_month'])

        # Wavelength axis for plotting (from Ed0 column names)
        wv_for_plot = extract_wavelengths_from_columns(Ed0, pattern='ed0')
        plot_ed_profiles(df=Ed_physic, wmo=wmo, kd_df=Kd,
                         wv_target=PLOT_TARGET_WAVELENGTHS,
                         wv_og=wv_for_plot, ed0=Ed0,
                         flags_df=flags_df_combined, depth_col='depth')

    # =====================================================================
    # Step 4 -- Re-plot one float's saved CSVs (sanity check)
    # =====================================================================
    last_wmo = wmos[-1] if wmos else None
    if last_wmo is not None:
        Kd = pd.read_csv(os.path.join(PROCESSED_PROFILES, last_wmo, f'{last_wmo}_Kd.csv'))
        Ed0 = pd.read_csv(os.path.join(PROCESSED_PROFILES, last_wmo, f'{last_wmo}_Ed0.csv'))
        Ed_all = pd.read_csv(os.path.join(PROCESSED_PROFILES, last_wmo, f'{last_wmo}_Ed.csv'))
        wavelengths = extract_wavelengths_from_columns(Ed0, pattern='ed0')

        filtered_df = Ed_all[Ed_all['depth'] <= 100]
        ed_columns = [c for c in filtered_df.columns if c.startswith('ed') and not c.startswith('ed0')]

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])
        colors = cm.viridis(np.linspace(0, 1, len(filtered_df)))
        for (_, row), color in zip(filtered_df.iterrows(), colors):
            ax1.plot(wavelengths, row[ed_columns].values, color=color, alpha=0.7)
        ax1.set_xlabel('Wavelength (nm)', fontsize=20)
        ax1.set_ylabel(r'$E_d$ (W m$^{-2}$ nm$^{-1}$)', fontsize=27)
        ax1.set_title(f'Float {last_wmo} $E_d$ Spectra', fontsize=26)
        ax1.tick_params(axis='both', which='major', labelsize=18)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Depth (m)', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1])
        specific_wavelengths = find_closest_wavelengths(
            [380.0, 440.0, 490.0, 555.0, 620.0], wavelengths)
        plot_colors = ['purple', 'indigo', 'lightblue', 'green', 'red']
        for i, (wavelength, plot_color) in enumerate(zip(specific_wavelengths, plot_colors)):
            ax = fig.add_subplot(gs_bottom[i])
            ed_column = f'ed{wavelength}'
            for _, row in filtered_df.iterrows():
                ax.scatter(row[ed_column], row['depth'], color=plot_color, alpha=0.7, marker='o')
            if i == 0:
                ax.set_ylabel('Depth (m)', fontsize=24)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.set_xlabel(f'$E_d$({wavelength})', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show(block=False)

        kd_columns = [c for c in Kd.columns if c.startswith('kd')
                      and 'unc' not in c and '_se' not in c and '_bincount' not in c]
        for _, row in Kd.iterrows():
            plt.plot(wavelengths, row[kd_columns].values, alpha=0.5)
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('Kd', fontsize=14)
        plt.title('Kd spectra', fontsize=16)
        plt.grid(True)
        plt.show(block=False)

        #%% Detect burst bining
        import xarray as xr
        import numpy as np
        import pandas as pd
        import glob
        import os
        import re

        # Match the values in your processing config
        BURST_BIN_DEPTH_RESOLUTION = 0.1
        BURST_SHALLOW_DEPTH_M = 15.0
        BURST_MIN_PER_BIN = 3

        ROOT = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve'

        def _decode(x, default=''):
            if isinstance(x, bytes): return x.decode().strip()
            if isinstance(x, str): return x.strip()
            return default
        def find_ed_n_prof(data):
            sp = data.STATION_PARAMETERS.values
            for n in range(sp.shape[0]):
                if 'DOWN_IRRADIANCE_SPECTRUM' in [_decode(p) for p in sp[n]]:
                    return n
            return None
        def cycle_is_burst(aux_path):
            """Return (was_burst, max_per_shallow_bin). max_per_shallow_bin = 0 if no data."""
            try:
                ds = xr.open_dataset(aux_path)
            except Exception:
                return None, None
            n = find_ed_n_prof(ds)
            if n is None:
                ds.close()
                return None, None
            pres = ds.PRES.sel(N_PROF=n).values
            ds.close()
            pres = pres[~np.isnan(pres)]
            if len(pres) == 0:
                return False, 0
            # Round depths to the bin grid, restrict to shallow layer
            shallow = pres[pres <= BURST_SHALLOW_DEPTH_M]
            if len(shallow) == 0:
                return False, 0
            binned = np.round(shallow / BURST_BIN_DEPTH_RESOLUTION) * BURST_BIN_DEPTH_RESOLUTION
            _, counts = np.unique(binned, return_counts=True)
            max_count = int(counts.max())
            return max_count >= BURST_MIN_PER_BIN, max_count


        # Walk every float
        records = []
        float_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, '*'))
                            if os.path.isdir(d) and re.match(r'^\d{7}$', os.path.basename(d)))

        for fd in float_dirs:
            wmo = os.path.basename(fd)
            aux_files = sorted(glob.glob(os.path.join(fd, 'profiles', '*_aux.nc')))
            if not aux_files:
                continue
            n_total = 0
            n_burst = 0
            burst_cycles = []
            for fn in aux_files:
                m = re.search(r"_([0-9]+).*_aux\.nc$", fn)
                if not m: continue
                cyc = int(m.group(1))
                is_burst, _ = cycle_is_burst(fn)
                if is_burst is None:
                    continue
                n_total += 1
                if is_burst:
                    n_burst += 1
                    burst_cycles.append(cyc)
            records.append({
                'wmo': wmo,
                'n_total_cycles': n_total,
                'n_burst_cycles': n_burst,
                'pct_burst': round(100 * n_burst / n_total, 1) if n_total else 0.0,
                'first_burst_cycle': burst_cycles[0] if burst_cycles else None,
                'last_burst_cycle': burst_cycles[-1] if burst_cycles else None,
            })
            print(f"  {wmo}: {n_burst}/{n_total} burst cycles")

        audit = pd.DataFrame(records).sort_values('pct_burst', ascending=False)
        print("\n=== Burst-binning audit ===")
        print(audit.to_string(index=False))

        audit.to_csv(os.path.join(ROOT, 'New_Outputs_NotRaw', 'burst_audit.csv'), index=False)
        print(f"\nSaved to {os.path.join(ROOT, 'New_Outputs_NotRaw', 'burst_audit.csv')}")