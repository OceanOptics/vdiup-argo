# Matchup our Kd matchups with PACE data. Retrive # of matchup and performance of Kd retrievals
import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats, odr
import re
import xarray as xr
import glob
import os
import concurrent.futures
import shutil
from collections import defaultdict



# Following is taken from PACE hackweek
# Satellite Matchup Constants
# Short names for earthaccess lookup
SAT_LOOKUP = {
    "PACE AOP": "PACE_OCI_L2_AOP",
    "PACE IOP": 'PACE_OCI_L2_IOP',
    "PACE KD": 'PACE_OCI_L3M_KD_NRT',
    "AQUA": "MODISA_L2_OC",
    "TERRA": "MODIST_L2_OC",
    "NOAA-20": "VIIRSJ1_L2_OC",
    "NOAA-21": "VIIRSJ2_L2_OC",
    "SUOMI-NPP": "VIIRSN_L2_OC"
    }
l2_flags_list = [
    "ATMFAIL", "LAND", "PRODWARN", "HIGLINT", "HILT", "HISATZEN", "COASTZ",
    "SPARE", "STRAYLIGHT", "CLDICE", "COCCOLITH", "TURBIDW", "HISOLZEN",
    "SPARE", "LOWLW", "CHLFAIL", "NAVWARN", "ABSAER", "SPARE", "MAXAERITER",
    "MODGLINT", "CHLWARN", "ATMWARN", "SPARE", "SEAICE", "NAVFAIL", "FILTER",
    "SPARE", "BOWTIEDEL", "HIPOL", "PRODFAIL", "SPARE"]
L2_FLAGS = {flag: 1 << idx for idx, flag in enumerate(l2_flags_list)}

# Bailey and Werdell 2006 exclusion criteria
EXCLUSION_FLAGS = ["LAND", "HIGLINT", "HILT", "STRAYLIGHT", "CLDICE",
                   "ATMFAIL", "LOWLW", "FILTER", "NAVFAIL", "NAVWARN"]

def get_fivebyfive(file, latitude, longitude, rrs_wavelengths, variable_wanted):
    """
    Get stats on a 5x5 box around station coordinates of a satellite granule.

    Parameters
    ----------
    file: earthaccess granule object
        Satellite granule from earthaccess.
    latitude: float
        In decimal degrees for Aeronet-OC site for matchups
    longitude: float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    rrs_wavelengths: numpy array
        Rrs wavelengths (from wavelength_3d for OCI)
    variable_wanted: str
        Variable to extract from the granule. Either 'Rrs' or 'Kd'

    Returns
    -------
    None.
    """
    with xr.open_dataset(file, group="navigation_data") as ds_nav:
        sat_lat = ds_nav['latitude'].values
        sat_lon = ds_nav['longitude'].values

    # Calculate the Euclidean distance for 2D lat/lon arrays
    distances = np.sqrt((sat_lat - latitude)**2 + (sat_lon - longitude)**2)

    # Find the index of the minimum distance
    # Dimensions are (lines, pixels)
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    center_line, center_pixel = min_dist_idx

    # Get indices for a 5x5 box around the center pixel
    line_start = max(center_line - 2, 0)
    line_end = min(center_line + 2 + 1, sat_lat.shape[0])
    pixel_start = max(center_pixel - 2, 0)
    pixel_end = min(center_pixel + 2 + 1, sat_lat.shape[1])

    # Extract the data
    with xr.open_dataset(file, group="geophysical_data") as ds_data:
        if variable_wanted == 'Rrs':
            rrs_data = ds_data['Rrs'].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end)
                ).values
        elif variable_wanted == 'Kd':
            rrs_data = ds_data['Kd'].isel(  number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end)
                ).values

        flags_data = ds_data['l2_flags'].isel(
            number_of_lines=slice(line_start, line_end),
            pixels_per_line=slice(pixel_start, pixel_end)
            ).values

    # Calculate the bitwise OR of all flags in EXCLUSION_FLAGS to get a mask
    exclude_mask = sum(L2_FLAGS[flag] for flag in EXCLUSION_FLAGS)

    # Create a boolean mask
    # True means the flag value does not contain any of the EXCLUSION_FLAGS
    valid_mask = np.bitwise_and(flags_data, exclude_mask) == 0

    # Get stats and averages
    if valid_mask.any():
        rrs_valid = rrs_data[valid_mask]
        rrs_std_initial = np.nanstd(rrs_valid, axis=0)
        rrs_mean_initial = np.nanmean(rrs_valid, axis=0)

        # Exclude spectra > 1.5 stdevs away
        std_mask = np.all(
            np.abs(rrs_valid - rrs_mean_initial) <= 1.5 * rrs_std_initial,
            axis=1)
        rrs_std = np.nanstd(rrs_valid[std_mask], axis=0)
        rrs_mean = np.nanmean(rrs_valid[std_mask], axis=0).flatten()

        # Matchup criteria uses cv as median of 405-570nm
        rrs_cv = rrs_std / rrs_mean
        rrs_cv_median = np.nanmedian(rrs_cv[(rrs_wavelengths >= 405)
                                         & (rrs_wavelengths <= 570)])
    else:
        rrs_cv_median = np.nan
        rrs_mean = np.nan * np.empty_like(rrs_wavelengths)

    # Put in dictionary of the row
    row = {
        "oci_datetime": pd.to_datetime(file.granule["umm"]["TemporalExtent"]
                                       ["RangeDateTime"]["BeginningDateTime"]),
        "oci_cv": rrs_cv_median,
        "oci_latitude": sat_lat[center_line, center_pixel],
        "oci_longitude": sat_lon[center_line, center_pixel],
        "oci_pixel_valid": np.sum(valid_mask)
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(rrs_wavelengths, rrs_mean):
        if variable_wanted =='Rrs':
            key = f'oci_rrs{int(wavelength)}'
            row[key] = mean_value
        elif variable_wanted == 'Kd':
            key = f'oci_kd{int(wavelength)}'
            row[key] = mean_value

    return row
def get_kd(file, latitude, longitude, Kd_wavelengths):
    """
    Get stats on the pixel around station coordinates of a satellite granule.

    Parameters
    ----------
    file : earthaccess granule object
        Satellite granule from earthaccess.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    rrs_wavelengths ; numpy array
        Rrs wavelengths (from wavelength_3d for OCI)

    Returns
    -------
    None.
    """
    with xr.open_dataset(file) as ds_nav:
        sat_lat = ds_nav['lat'].values
        sat_lon = ds_nav['lon'].values

    sat_lat = np.tile(sat_lat[:, np.newaxis], (1, 8640))
    sat_lon = np.tile(sat_lon, (4320, 1))

    # Calculate the Euclidean distance for 2D lat/lon arrays
    distances = np.sqrt((sat_lat - latitude)**2 + (sat_lon - longitude)**2)

    # Find the index of the minimum distance
    # Dimensions are (lines, pixels)
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    center_line, center_pixel = min_dist_idx

    # Extract the data
    with xr.open_dataset(file) as ds_data:
        kd_data = ds_data['Kd'].isel(
            lat=center_line,
            lon=center_pixel
            ).values

    # Get stats and averages
    #kd_std_initial = np.std(kd_data, axis=0)
    kd_mean_initial = kd_data

    # Put in dictionary of the row
    row = {
        "oci_datetime": pd.to_datetime(file.granule["umm"]["TemporalExtent"]
                                       ["RangeDateTime"]["BeginningDateTime"]),
        "oci_latitude": sat_lat[center_line, center_pixel],
        "oci_longitude": sat_lon[center_line, center_pixel],
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(Kd_wavelengths, kd_mean_initial):
        key = f'oci_kd{int(wavelength)}'
        row[key] = mean_value

    return row
def process_granule(file, latitude, longitude, rrs_wavelengths, variable_wanted):
    granule_date = pd.to_datetime(file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    print(f"Running Granule: {granule_date}")
    return get_fivebyfive(file, latitude, longitude, rrs_wavelengths, variable_wanted)
def process_satellite_rrs(kd_loc, variable_wanted, sat="PACE"):
    if sat not in SAT_LOOKUP.keys():
        raise ValueError(f"{sat} is not in the lookup dictionary. Available sats are: {', '.join(SAT_LOOKUP)}")
    short_name = SAT_LOOKUP[sat]

    all_rows = []
    rrs_wavelengths = None

    try:
        for idx, row in kd_loc.iterrows():
            #date = pd.to_datetime(row['date'], format='%Y%m%d').strftime("%Y-%m-%d")
            date = pd.to_datetime(row['date'], format='%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")
            latitude = row['lat']
            longitude = row['lon']
            time_bounds = (f"{date}T00:00:00Z", f"{date}T23:59:59Z")

            try:
                results = earthaccess.search_data(temporal=time_bounds, point=(longitude, latitude), short_name=short_name)
            except IndexError:
                print(f"No data found for {date}, {latitude}, {longitude}")
                continue

            files = earthaccess.open(results)

            if rrs_wavelengths is None:
                rrs_wavelengths = extract_rrs_wavelengths(files[0])

            for file in files:
                try:
                    row_data = process_granule(file, latitude, longitude, rrs_wavelengths, variable_wanted)
                    all_rows.append(row_data)
                except Exception as e:
                    print(f"Error processing {date}, {latitude}, {longitude}: {e}")
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        return pd.DataFrame(all_rows)
def extract_rrs_wavelengths(file):
    """Extract wavelengths — handles both full hyperspectral (AOP) and
    17-band IOP products automatically."""
    with xr.open_dataset(file, group="sensor_band_parameters") as ds:
        print("sensor_band_parameters variables:", list(ds.variables))
        # IOP product uses 'wavelength' or similar; AOP uses 'wavelength_3d'
        if 'wavelength_3d' in ds.variables:
            return ds['wavelength_3d'].values
        elif 'wavelength' in ds.variables:
            return ds['wavelength'].values
        else:
            # Fallback: infer from the Kd variable's third dimension
            with xr.open_dataset(file, group="geophysical_data") as dg:
                n_wl = dg['Kd'].shape[2]
            raise ValueError(
                f"Cannot find wavelength variable. Kd has {n_wl} bands. "
                f"Available: {list(ds.variables)}")
def gather_search_results(kd_loc, sat):
    """Search once per unique (date, rough lat/lon). Returns [(profile_idx, granule), ...]."""
    short_name = SAT_LOOKUP[sat]
    kd_loc = kd_loc.copy()
    kd_loc['_date_str'] = pd.to_datetime(kd_loc['date']).dt.strftime('%Y-%m-%d')
    kd_loc['_lat_r'] = kd_loc['lat'].round(1)   # ~10 km grouping; granules are much bigger
    kd_loc['_lon_r'] = kd_loc['lon'].round(1)

    cache = {}
    pairs = []
    for idx, row in kd_loc.iterrows():
        key = (row['_date_str'], row['_lat_r'], row['_lon_r'])
        if key not in cache:
            try:
                cache[key] = earthaccess.search_data(
                    temporal=(f"{row['_date_str']}T00:00:00Z",
                              f"{row['_date_str']}T23:59:59Z"),
                    point=(row['lon'], row['lat']),
                    short_name=short_name)
            except Exception as e:
                print(f"  search failed for {key}: {e}")
                cache[key] = []
        for granule in cache[key]:
            pairs.append((idx, granule))
    return pairs
def process_granule_multi_local(local_path, granule, profile_specs,
                                 rrs_wavelengths, variable_wanted):
    """Same as process_granule_multi but operates on a local .nc path."""
    try:
        ds_nav = xr.open_dataset(local_path, group="navigation_data")
        ds_geo = xr.open_dataset(local_path, group="geophysical_data")
    except Exception as e:
        print(f"  open failed: {e}")
        return []

    granule_time = pd.to_datetime(
        granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    exclude_mask = sum(L2_FLAGS[f] for f in EXCLUSION_FLAGS)
    prefix = 'oci_kd' if variable_wanted == 'Kd' else 'oci_rrs'

    try:
        # Now that the file is local, just read full arrays — it's all memory.
        sat_lat = ds_nav['latitude'].values
        sat_lon = ds_nav['longitude'].values
        var_data = ds_geo[variable_wanted].values
        flags_data = ds_geo['l2_flags'].values

        out = []
        for prof_idx, lat, lon in profile_specs:
            distances = (sat_lat - lat) ** 2 + (sat_lon - lon) ** 2
            cl, cp = np.unravel_index(np.argmin(distances), distances.shape)
            ls = max(cl - 2, 0); le = min(cl + 3, sat_lat.shape[0])
            ps = max(cp - 2, 0); pe = min(cp + 3, sat_lat.shape[1])
            v = var_data[ls:le, ps:pe]
            f = flags_data[ls:le, ps:pe]
            valid = np.bitwise_and(f, exclude_mask) == 0

            # v is (≤5, ≤5, n_wl) for 3D products (IOP) or (≤5, ≤5) for 2D.
            # Normalise to 3D so the rest of the logic is identical.
            if v.ndim == 2:
                v = v[:, :, np.newaxis]
            n_wl = v.shape[2]

            if valid.any():
                v_valid = v[valid, :]  # (n_valid_px, n_wl)
                mean0 = np.nanmean(v_valid, axis=0)
                std0 = np.nanstd(v_valid, axis=0)
                sel = np.all(np.abs(v_valid - mean0) <= 1.5 * std0, axis=1)
                v_final = v_valid[sel, :]
                n_valid_final = int(sel.sum())
                mean = np.nanmean(v_final, axis=0).flatten()  # (n_wl,)
                cv = np.nanstd(v_final, axis=0) / np.where(mean == 0, np.nan, mean)
                cv_med = np.nanmedian(
                    cv[(rrs_wavelengths >= 405) & (rrs_wavelengths <= 570)])
            else:
                cv_med = np.nan
                n_valid_final = 0
                mean = np.full(len(rrs_wavelengths), np.nan, dtype=float)

            row = {
                'oci_datetime': granule_time,
                'oci_cv': cv_med,
                'oci_latitude': sat_lat[cl, cp],
                'oci_longitude': sat_lon[cl, cp],
                'oci_pixel_valid': n_valid_final,  # now post-sigma filter
            }

            for wv, mv in zip(rrs_wavelengths, mean):
                row[f'{prefix}{int(wv)}'] = mv
            out.append(row)
    except Exception as e:
        print(f"  read failed: {e}")
        out = []
    finally:
        ds_nav.close()
        ds_geo.close()

    return out

def process_satellite_rrs_fast(kd_loc, variable_wanted, sat="PACE IOP",
                               max_workers=4, download_dir=None,
                               checkpoint_path=None, checkpoint_every=50):
    """Two-phase: download all granules to disk in parallel, then process locally.

    download_dir: directory to cache .nc files (created if missing).
    checkpoint_path: CSV path to save partial results; enables resume.
    """
    import os, time
    if download_dir is None:
        download_dir = os.path.expanduser("~/pace_granule_cache")
    os.makedirs(download_dir, exist_ok=True)

    # ---- Phase 1: search + dedup ----
    print(f"  searching granules for {len(kd_loc)} profiles...")
    pairs = gather_search_results(kd_loc, sat)
    by_granule = defaultdict(list)
    granule_objs = {}
    for prof_idx, granule in pairs:
        gid = granule["meta"]["concept-id"]
        by_granule[gid].append(prof_idx)
        granule_objs[gid] = granule
    print(f"  {len(pairs)} profile-granule pairs → {len(by_granule)} unique granules")

    # ---- Resume: load checkpoint ----
    done_gids = set()
    existing_rows = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        if '_gid' in ckpt.columns:
            done_gids = set(ckpt['_gid'].unique())
            existing_rows = ckpt.to_dict('records')
            print(f"  resuming: {len(done_gids)} granules already in checkpoint")
    todo_gids = [g for g in by_granule if g not in done_gids]
    print(f"  {len(todo_gids)} granules to process")
    if not todo_gids:
        return pd.DataFrame(existing_rows).drop(columns=['_gid'], errors='ignore')

    # ---- Phase 2: bulk download with earthaccess ----
    print(f"  downloading granules to {download_dir}...")
    t0 = time.time()
    granules_to_download = [granule_objs[g] for g in todo_gids]
    # earthaccess.download handles parallelism, retries, resumption internally
    local_paths = earthaccess.download(granules_to_download, download_dir, threads=8)
    # Map back: gid -> local path
    gid_to_path = dict(zip(todo_gids, local_paths))
    print(f"  download done in {(time.time() - t0) / 60:.1f} min "
          f"({len(local_paths)} files, {(time.time() - t0) / len(local_paths):.1f} s/file)")

    # ---- Phase 3: process local files (fast, parallel, no network) ----
    rrs_wavelengths = extract_rrs_wavelengths(local_paths[0])

    def _job(gid):
        path = gid_to_path[gid]
        granule = granule_objs[gid]
        specs = [(idx, kd_loc.loc[idx, 'lat'], kd_loc.loc[idx, 'lon'])
                 for idx in by_granule[gid]]
        rows = process_granule_multi_local(path, granule, specs,
                                           rrs_wavelengths, variable_wanted)
        for r in rows:
            r['_gid'] = gid
        return rows

    all_rows = list(existing_rows)
    n_done = n_error = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, g) for g in todo_gids]
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_rows.extend(fut.result(timeout=30))
            except Exception as e:
                n_error += 1
                if n_error <= 5:
                    print(f"  process failed: {e}")
            n_done += 1
            if checkpoint_path and n_done % checkpoint_every == 0:
                pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
                rate = n_done / (time.time() - t0)
                eta = (len(todo_gids) - n_done) / rate / 60 if rate > 0 else 0
                print(f"  {n_done}/{len(todo_gids)} processed  "
                      f"({rate:.1f}/s, ETA {eta:.1f} min, errors: {n_error})")

    if checkpoint_path:
        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
    print(f"  done: {len(all_rows)} rows, {n_error} errors")
    return pd.DataFrame(all_rows).drop(columns=['_gid'], errors='ignore')
def process_satellite_kd(kd_loc, sat="PACE"):
    """
    Download and process satellite data for matchups.

    Parameters
    ----------
    kd_loc : pandas DataFrame
        DataFrame containing columns 'date', 'lat', 'lon' for each matchup.
    sat : str
        Name of satellite to search. Must be in SAT_LOOKUP dict constant.

    Returns
    -------
    pandas DataFrame object
        Flattened table of all satellite granule matchups.
    """
    # Look up short name from constants
    if sat not in SAT_LOOKUP.keys():
        raise ValueError(f"{sat} is not in the lookup dictionary. Available "
                         f"sats are: {', '.join(SAT_LOOKUP)}")
    #short_name = SAT_LOOKUP[sat]

    # Initialize list to store results
    all_rows = []
    # Loop through each row in kd_loc
    for idx, row in kd_loc.iterrows():
        date = pd.to_datetime(row['date'], format='%Y%m%d').strftime("%Y-%m-%d")
        latitude = row['lat']
        longitude = row['lon']
        time_bounds = (
            f"{date}T00:00:00Z",
            f"{date}T23:59:59Z"
        )

            # Run Earthaccess data search
        try:
            results = earthaccess.search_data(temporal=time_bounds,
                                              point=(longitude, latitude),
                                              short_name='PACE_OCI_L3M_KD_NRT')
        except IndexError:
            print(f"No data found for {date}, {latitude}, {longitude}")
            continue

        # Filter the results before opening the files
        filtered_results = [result for result in results if 'DAY' in str(result) and '4km' in str(result)]

    # Open only the filtered results
        files = earthaccess.open(filtered_results)

        if not files:
            print(f"No 4km daytime granules found for {date}, {latitude}, {longitude}")
            continue

        # Pull out Rrs wavelengths for easier processing
        with xr.open_dataset(files[0]) as ds_bands:
            Kd_wavelengths = ds_bands["wavelength"].values

        # Loop through files and process
        for file in files:
            granule_date = pd.to_datetime(file.granule["umm"]["TemporalExtent"]
                                          ["RangeDateTime"]["BeginningDateTime"])
            print(f"Running Granule: {granule_date}")
            row_data = get_kd(file, latitude, longitude, Kd_wavelengths)
            all_rows.append(row_data)

    return pd.DataFrame(all_rows)
def match_data(df_sat, df_aoc,  cv_max=0.15,min_valid_pixels=10,    max_time_diff=180,  sza_max=70.0):
    """
    Match satellite and in-situ data following Bailey & Werdell (2006).
    Criteria applied:
      1. CV(405-570 nm) <= cv_max  (scene homogeneity) NOT FOR KD
      2. oci_pixel_valid >= min_valid_pixels  (>= 10 of 25 pixels)
      3. |time difference| <= max_time_diff minutes
      4. Solar zenith angle <= sza_max degrees  (daytime only)
    """
    from pysolar.solar import get_altitude

    df_sat = df_sat.copy()
    df_aoc = df_aoc.copy()
    df_sat['oci_datetime'] = pd.to_datetime(df_sat['oci_datetime']).dt.tz_localize(None)
    df_aoc['date']         = pd.to_datetime(df_aoc['date']).dt.tz_localize(None)

    time_window = pd.Timedelta(minutes=max_time_diff)

    # ── Satellite-side filters ──────────────────────────────────────────────
    n0 = len(df_sat)
    # # 1. CV threshold
    # df_sat_f = df_sat_f[df_sat['oci_cv'] <= cv_max]
    # print(f"  CV filter (≤{cv_max}):          {n0} → {len(df_sat_f)} sat rows")
    #No CV threshold for Kd
    df_sat_f=df_sat.copy()
    # 2. Minimum valid pixels
    df_sat_f = df_sat_f[df_sat_f['oci_pixel_valid'] >= min_valid_pixels]
    print(f"  Min pixels (≥{min_valid_pixels}):       {len(df_sat_f)} sat rows remaining")

    # ── In-situ-side filter: solar zenith ──────────────────────────────────
    def compute_sza(row):
        try:
            timestamp_utc = row['date'].tz_localize('UTC').to_pydatetime()
            solar_elevation = get_altitude(row['lat'], row['lon'], timestamp_utc)
            return 90.0 - solar_elevation
        except Exception:
            return 90.0

    df_aoc['_sza'] = df_aoc.apply(compute_sza, axis=1)
    n_aoc0 = len(df_aoc)
    df_aoc_f = df_aoc[df_aoc['_sza'] <= sza_max]
    print(f"  SZA filter  (≤{sza_max}°):       {n_aoc0} → {len(df_aoc_f)} in-situ rows")

    # ── Spatial + temporal matching ────────────────────────────────────────
    records = []
    for _, sat_row in df_sat_f.iterrows():
        oci_dt  = sat_row['oci_datetime']
        td      = (df_aoc_f['date'] - oci_dt).abs()
        dlat    = (df_aoc_f['lat'] - sat_row['oci_latitude']).abs()
        dlon    = (df_aoc_f['lon'] - sat_row['oci_longitude']).abs()
        matches = df_aoc_f[(td <= time_window) & (dlat <= 0.2) & (dlon <= 0.2)]
        if matches.empty:
            continue
        best_idx  = td[matches.index].idxmin()
        best      = matches.loc[best_idx]
        rec       = {**best.to_dict(), **sat_row.to_dict()}
        rec['time_diff'] = td[best_idx]
        rec['sza']       = best['_sza']
        records.append(rec)

    df_match = pd.DataFrame(records)
    # Drop the working column
    df_match = df_match.drop(columns=['_sza'], errors='ignore')
    print(f"\n  Final matchups: {len(df_match)}")
    return df_match
def match_data_kd(df_sat, df_aoc):
    """Create matchup dataframe based on selection criteria.

    Parameters
    ----------
    df_sat : pandas dataframe
        Satellite data from flat validation file.
    df_aoc : pandas dataframe
        Field data from flat validation file.
    -------
    pandas dataframe of matchups for product
    """
    # Setup
    df_match_list = []

    # Ensure both datetime columns are timezone-naive
    df_aoc['date'] = pd.to_datetime(df_aoc['date']).dt.tz_localize(None)
    df_sat['oci_datetime'] = pd.to_datetime(df_sat['oci_datetime']).dt.tz_localize(None)

    # Filter Field data based on Solar Zenith
    df_aoc_filtered = df_aoc

    # Filter satellite data based on cv threshold and on if there is more than 1 oci_pixel_value
    df_sat_filtered = df_sat[df_sat['oci_cv'] <= 0.15]
    df_sat_filtered = df_sat_filtered[df_sat_filtered['oci_pixel_valid'] != 0]

    for _, sat_row in df_sat_filtered.iterrows():
        # Filter field data based on time difference and coordinates
        within_lat = 0.2 >= abs(  df_aoc_filtered['lat'] - sat_row['oci_latitude'])
        within_lon = 0.2 >= abs(df_aoc_filtered['lon'] - sat_row['oci_longitude'])
        field_matches = df_aoc_filtered[within_lat & within_lon]

        if not field_matches.empty:
            # Select the best match based on time delta
            time_diff = abs(
                field_matches['date']-sat_row['oci_datetime'])
            best_match = field_matches.loc[time_diff.idxmin()]
            df_match_list.append({**best_match.to_dict(), **sat_row.to_dict()})

    df_match = pd.DataFrame(df_match_list)
    return df_match

# %%

# Look for the name of variable we want.
results = earthaccess.search_datasets(
    instrument="PACE",
keyword = 'AOP')
set((i.summary()["short-name"] for i in results))

# First load our complete list of Kd files.
directory = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw'
# Find all CSV files matching the pattern "*_Kd.csv"
csv_files = glob.glob(os.path.join(directory, '**', '*_Kd.csv'), recursive=True)

# Initialize an empty list to store the data

data = []

all_columns = set()

# First pass to collect all unique kd columns
for file in csv_files:
    kd_file = pd.read_csv(file)
    kd_columns = [col for col in kd_file.columns if re.match(r'kd\d+\.0$', col) ]
    #or re.match(r'kd\d+\.0_unc$', col)
    all_columns.update(kd_columns)

# Ensure all DataFrames have the same columns
for file in csv_files:
    # Extract the WMO from the filename
    wmo = os.path.basename(file).split('_Kd.csv')[0]

    kd_file = pd.read_csv(file)
    temp_df = pd.DataFrame()
    temp_df['date'] = pd.to_datetime(kd_file['date'].astype(str) + ' ' + kd_file['time'])
    temp_df['lat'] = kd_file['lat']
    temp_df['lon'] = kd_file['lon']
    temp_df['WMO'] = [f"{wmo}_{str(profile).zfill(3)}" for profile in kd_file['profile']]
    temp_df['quality'] = kd_file['quality']

    # Include all columns matching kdXXX.0 and kdXXX.0_unc
    for col in all_columns:
        temp_df[col] = kd_file[col] if col in kd_file.columns else np.nan

    # Append the DataFrame to the list
    data.append(temp_df)

# Concatenate all DataFrames
kd_loc = pd.concat(data, ignore_index=True)
kd_loc['date'] = pd.to_datetime(kd_loc['date'], format='%Y%m%d')

kd_loc = kd_loc[kd_loc['date'] >= '2024-04-01']

# Only keep profiles with QC of 2
kd_loc_all = kd_loc.copy()
kd_loc = kd_loc[kd_loc['quality'] != 2]

sat_rrs = process_satellite_rrs_fast(kd_loc, sat="PACE IOP", variable_wanted='Kd',
    max_workers=4, download_dir='/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/pace_cache',
    checkpoint_path='/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_checkpoint.csv')
sat_rrs.to_csv('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_final.csv')
#
# cache_dir = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/pace_cache'
# ckpt = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_checkpoint.csv'
# shutil.rmtree(cache_dir)
# os.makedirs(cache_dir)
# print(f"Cache cleared.")
# if os.path.exists(ckpt):
#     os.remove(ckpt)
#     print("Checkpoint cleared.")
# cache_dir = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/pace_cache'
# ckpt = '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_checkpoint.csv'
#

# Compute matchups
matchups = match_data(sat_rrs, kd_loc, cv_max=0.8, max_time_diff=380, sza_max=70.0)
matchups = matchups.dropna(axis=1, how='all')
#save to csv
matchups.to_csv('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/matchups_PACE_Kd_L2.csv')

#Drop the rows where quality is 2
matchups_clean = matchups[matchups.quality != 2]
matchups_clean = matchups_clean.reset_index(drop=True)

# Define a function to extract wavelengths and values
def extract_wavelengths_and_values(row, pattern):
    columns = [col for col in row.index if re.match(pattern, col)]
    wavelengths = [int(re.search(pattern, col).group(1)) for col in columns]
    values = row[columns].values
    return wavelengths, values

# Define colors and symbols
colors = plt.cm.viridis(np.linspace(0, 1, len(matchups_clean)))
kd_symbol = 'o'
rrs_symbol = 's'

# Plot each row
for idx, row in matchups_clean.iterrows():
    kd_wavelengths, kd_values = extract_wavelengths_and_values(row, r'kd(\d+)\.0$')
    rrs_wavelengths, rrs_values = extract_wavelengths_and_values(row, r'oci_kd(\d+)$')
    plt.plot(kd_wavelengths, kd_values, kd_symbol, color=colors[idx])
    plt.plot(rrs_wavelengths, rrs_values, rrs_symbol, color=colors[idx])

# Add labels and title
plt.xlabel('Wavelength (nm)')
plt.ylabel('Values')
plt.title('Kd and OCI Kd vs Wavelengths')
plt.grid(True)
plt.show()
#$$



# RElatve diff plot
# Define colors and symbols
colors = plt.cm.viridis(np.linspace(0, 1, len(matchups_clean)))
all_kd_values = []
all_oci_kd_values = []
all_wavelengths = []

# Plot each row
for idx, row in matchups_clean.iterrows():
    row = row.dropna().sort_index()
    kd_wavelengths, kd_values = extract_wavelengths_and_values(row, r'kd(\d+)\.0$')
    if kd_values.size == 0:
        print('empty')
        continue
    oci_kd_wavelengths, oci_kd_values = extract_wavelengths_and_values(row, r'oci_kd(\d+)$')

    kd_wavelengths = np.array(kd_wavelengths, dtype=float)
    kd_values = np.array(kd_values, dtype=float)
    oci_kd_wavelengths = np.array(oci_kd_wavelengths, dtype=float)
    oci_kd_values = np.array(oci_kd_values, dtype=float)

    interpolated_kd_values = np.interp(oci_kd_wavelengths, kd_wavelengths, kd_values)
    relative_difference = np.abs(oci_kd_values - interpolated_kd_values) / (
                (oci_kd_values + interpolated_kd_values) / 2) * 100

    # Collect data for the scatter plot
    all_kd_values.extend(interpolated_kd_values)
    all_oci_kd_values.extend(oci_kd_values)
    all_wavelengths.extend(oci_kd_wavelengths)

    plt.plot(oci_kd_wavelengths, relative_difference, color=colors[idx])

# Add labels and title
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Difference (%)')
plt.title('Relative Difference between OCI Kd and Kd')
plt.grid(True)
plt.show()


# Create the scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(all_kd_values, all_oci_kd_values, c=all_wavelengths, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label='Wavelength (nm)')
plt.plot([0, 2], [0, 2], 'r--', label='1:1 Line')
plt.xlabel('Argo in-situ Kd($\lambda$)', fontsize=18)
plt.ylabel('PACE OCI Kd($\lambda$) ', fontsize=18)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.savefig('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/Scatter_Kd_PACE_vs_Float.png')
plt.show()

matchups.to_csv('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/Outputs/matchup_loc_withPACE.csv')

#%% Make a nice map of the location of the matchups for slidesimport cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

proj = ccrs.Mollweide()

fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': proj})

ax.set_global()
ax.add_feature(cfeature.LAND, color='#e8e8e8', zorder=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='#999999', linewidth=0.4, zorder=2)
ax.add_feature(cfeature.BORDERS, edgecolor='#999999', linewidth=0.2, zorder=2)
ax.spines['geo'].set_edgecolor('black')
ax.spines['geo'].set_linewidth(1.5)


ax.scatter(kd_loc_all['lon'], kd_loc_all['lat'],
           color='#aaaaaa', s=60, alpha=0.65,
           edgecolor='white', linewidth=0.4, zorder=3,
           transform=ccrs.PlateCarree())

ax.scatter(kd_loc['lon'], kd_loc['lat'],
           color='#2171b5', s=60, alpha=0.65,
           edgecolor='white', linewidth=0.4, zorder=3,
           transform=ccrs.PlateCarree())

ax.scatter(matchups['lon'], matchups['lat'],
           color='#e63946', s=200, alpha=0.95,
           edgecolor='white', linewidth=0.6, zorder=4,
           marker='*', transform=ccrs.PlateCarree())

leg_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
               markersize=20, alpha=0.8, label='All BGC-Argo profiles'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2171b5',
               markersize=20, alpha=0.8, label='BGC-Argo profiles passing QC'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#e63946',
               markersize=30, label='PACE L2 Kd matchups'),
]
ax.legend(handles=leg_handles, fontsize=20, loc='lower left',
          framealpha=0.9, edgecolor='#cccccc', frameon=True)

plt.tight_layout()
plt.savefig(
    '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/'
    'New_Outputs_NotRaw/Map_Location_Profiles.png',
    dpi=250, bbox_inches='tight')
plt.show()


#%% import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
from scipy import stats

# ── helpers ──────────────────────────────────────────────────────────────────
def extract_matched_pairs(matchups_clean):
    """Return arrays: float_kd, pace_kd, wavelength — one entry per (profile, wavelength)."""
    float_kd, pace_kd, wls = [], [], []
    for _, row in matchups_clean.iterrows():
        for col in row.index:
            m = re.match(r'kd(\d+)\.0$', col)
            if not m:
                continue
            wl = int(m.group(1))
            oci_col = f'oci_kd{wl}'
            if oci_col not in row.index:
                continue
            f = row[col]
            p = row[oci_col]
            if np.isnan(f) or np.isnan(p) or f <= 0 or p <= 0:
                continue
            float_kd.append(f)
            pace_kd.append(p)
            wls.append(wl)
    return np.array(float_kd), np.array(pace_kd), np.array(wls)
def type2_regression(x, y):
    """Geometric mean (type-II) regression on log-log data. Returns slope, intercept."""
    lx, ly = np.log10(x), np.log10(y)
    slope_ols = stats.linregress(lx, ly).slope
    slope = np.sign(slope_ols) * (np.std(ly) / np.std(lx))
    intercept = np.mean(ly) - slope * np.mean(lx)
    return slope, intercept


fig, ax = plt.subplots(figsize=(9, 7))

# Interpolate Argo onto PACE wavelengths
profile_curves = []
for _, row in matchups_clean.iterrows():
    argo_wls, argo_vals = extract_wavelengths_and_values(row, r'kd(\d+)\.0$')
    pace_wls, pace_vals = extract_wavelengths_and_values(row, r'oci_kd(\d+)$')

    argo_wls = np.array(argo_wls, dtype=float)
    argo_vals = np.array(pd.to_numeric(argo_vals, errors='coerce'), dtype=float)
    pace_wls = np.array(pace_wls, dtype=float)
    pace_vals = np.array(pd.to_numeric(pace_vals, errors='coerce'), dtype=float)
    argo_ok = ~np.isnan(argo_vals)
    pace_ok = ~np.isnan(pace_vals)
    if argo_ok.sum() < 3 or pace_ok.sum() < 3:
        continue

    # Sort by wavelength before interpolating
    argo_sort = np.argsort(argo_wls[argo_ok])
    argo_wls_sorted = argo_wls[argo_ok][argo_sort]
    argo_vals_sorted = argo_vals[argo_ok][argo_sort]

    argo_interp = np.interp(pace_wls, argo_wls_sorted, argo_vals_sorted,
                            left=np.nan, right=np.nan)

    both_ok = pace_ok & ~np.isnan(argo_interp) & (argo_interp > 0) & (pace_vals > 0)
    if both_ok.sum() < 3:
        continue

    rel_diff = (pace_vals[both_ok] - argo_interp[both_ok]) / argo_interp[both_ok] * 100
    profile_curves.append((pace_wls[both_ok], rel_diff))
# ── Individual profile curves in light gray ───────────────────────────────
for wls_used, rd in profile_curves:
    ax.plot(wls_used, rd, color='lightgray', lw=0.8, alpha=0.6, zorder=1)

# ── Per-wavelength median ± IQR ───────────────────────────────────────────
rd_by_wl = {wl: [] for wl in pace_wls}
for wls_used, rd in profile_curves:
    for wl, v in zip(wls_used, rd):
        rd_by_wl[wl].append(v)

wl_plot  = np.array([wl for wl in pace_wls if len(rd_by_wl[wl]) >= 3])
medians  = np.array([np.median(rd_by_wl[wl]) for wl in wl_plot])
q25      = np.array([np.percentile(rd_by_wl[wl], 25) for wl in wl_plot])
q75      = np.array([np.percentile(rd_by_wl[wl], 75) for wl in wl_plot])

ax.axhline(0, color='k', lw=1, ls='--', zorder=2)
ax.fill_between(wl_plot, q25, q75, alpha=0.35, color='steelblue', zorder=3, label='Interquartile')
ax.plot(wl_plot, medians, color='steelblue', lw=2.5, zorder=4, label='Median bias')

ax.set_xlabel('Wavelength (nm)', fontsize=15)
ax.set_ylabel('Relative Difference (%)', fontsize=15)
ax.set_ylim(-100, 150)   # or whatever range shows the bulk of the data
ax.legend(fontsize=14, loc='upper right')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(
    '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/'
    'New_Outputs_NotRaw/Spectral_Bias_ribbon.png', dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(9, 8))

float_kd = np.array(all_kd_values)
pace_kd  = np.array(all_oci_kd_values)
wls      = np.array(all_wavelengths)
rel_diff = (pace_kd - float_kd) / float_kd * 100

unique_wls   = np.sort(np.unique(wls))
colors_rgb   = np.array([wavelength_to_rgb(w) for w in wls])
unique_colors= np.array([wavelength_to_rgb(w) for w in unique_wls])
wl_min, wl_max = wls.min(), wls.max()
spec_cmap = mcolors.LinearSegmentedColormap.from_list(
    'spectrum', list(zip(
        (unique_wls - wl_min) / (wl_max - wl_min),
        unique_colors)))
norm = mcolors.Normalize(vmin=wl_min, vmax=wl_max)

sc = ax.scatter(float_kd, pace_kd, c=colors_rgb,
                alpha=0.75, edgecolor='none', s=40, zorder=3)
cb = plt.colorbar(mcm.ScalarMappable(norm=norm, cmap=spec_cmap), ax=ax)
cb.set_label('Wavelength (nm)', fontsize=18)
cb.ax.tick_params(labelsize=15)

lims = [min(float_kd.min(), pace_kd.min()) * 0.8,
        max(float_kd.max(), pace_kd.max()) * 1.2]
ax.plot(lims, lims, 'k--', lw=1.4, label='1:1', zorder=2)

slope, intercept = type2_regression(float_kd, pace_kd)
x_fit = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 200)
y_fit = 10 ** (intercept + slope * np.log10(x_fit))
ax.plot(x_fit, y_fit, color='dimgray', lw=1.8,
        label=f'Type-II (slope={slope:.2f})', zorder=2)

r2   = np.corrcoef(np.log10(float_kd), np.log10(pace_kd))[0, 1] ** 2
bias = np.median(rel_diff)
rmse = np.sqrt(np.mean((np.log10(pace_kd) - np.log10(float_kd)) ** 2))
n    = len(float_kd)
stats_txt = (f'N = {n}\n'
             f'R² = {r2:.3f}\n'
             f'Bias = {bias:+.1f}%\n'
             f'Log RMSE = {rmse:.3f}')
ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
        va='top', ha='left', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85,
                  edgecolor='#cccccc'))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims);    ax.set_ylim(lims)
ax.set_xlabel('Argo in-situ Kd(λ)', fontsize=20)
ax.set_ylabel('PACE OCI Kd(λ)', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=14, loc='lower right')
ax.grid(True, which='both', alpha=0.25)
plt.tight_layout()
plt.savefig(
    '/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/'
    'New_Outputs_NotRaw/Scatter_Kd_polished.png', dpi=200)
plt.show()