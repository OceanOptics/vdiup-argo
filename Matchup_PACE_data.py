# Matchup our Kd matchups with PACE data. Retrive # of matchup and performance of Kd retrievals
import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pvlib.solarposition as sunpos
from scipy import stats, odr
import re
import xarray as xr
import glob
import os
import concurrent.futures
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
    with xr.open_dataset(file, group="sensor_band_parameters") as ds_bands:
        return ds_bands["wavelength_3d"].values
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
def point_in_granule_bbox(granule, lat, lon):
    """Quick spatial check using CMR bounding rectangle/polygon metadata."""
    try:
        spatial = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if "BoundingRectangles" in spatial:
            for box in spatial["BoundingRectangles"]:
                if (box["SouthBoundingCoordinate"] <= lat <= box["NorthBoundingCoordinate"]
                    and box["WestBoundingCoordinate"] <= lon <= box["EastBoundingCoordinate"]):
                    return True
            return False
    except (KeyError, TypeError):
        pass
    return True
def process_granule_multi(granule, profile_specs, rrs_wavelengths, variable_wanted):
    """Process one granule for all profiles that need it.
    profile_specs = [(idx, lat, lon), ...]
    Uses lazy reads: coarse subsample to find the rough pixel, then a small
    full-resolution window around it, instead of pulling the whole lat/lon/data arrays.
    """
    try:
        file = earthaccess.open([granule])[0]
        ds_nav = xr.open_dataset(file, group="navigation_data")
        ds_geo = xr.open_dataset(file, group="geophysical_data")
    except Exception as e:
        print(f"  open failed: {e}")
        return []

    granule_time = pd.to_datetime(
        granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    exclude_mask = sum(L2_FLAGS[f] for f in EXCLUSION_FLAGS)
    prefix = 'oci_kd' if variable_wanted == 'Kd' else 'oci_rrs'

    try:
        # Step 1: coarse subsample for cheap nearest-pixel search (one-time per granule)
        sat_lat_coarse = ds_nav['latitude'][::8, ::8].values
        sat_lon_coarse = ds_nav['longitude'][::8, ::8].values
        nl_full, np_full = ds_nav['latitude'].shape

        out = []
        for prof_idx, lat, lon in profile_specs:
            # Step 2: rough nearest on the coarse grid
            d = (sat_lat_coarse - lat) ** 2 + (sat_lon_coarse - lon) ** 2
            ci, cj = np.unravel_index(np.argmin(d), d.shape)

            # Step 3: full-resolution window around the coarse hit
            cl_lo = max(ci * 8 - 8, 0); cl_hi = min(ci * 8 + 8, nl_full)
            cp_lo = max(cj * 8 - 8, 0); cp_hi = min(cj * 8 + 8, np_full)
            lat_win = ds_nav['latitude'].isel(
                number_of_lines=slice(cl_lo, cl_hi),
                pixels_per_line=slice(cp_lo, cp_hi)).values
            lon_win = ds_nav['longitude'].isel(
                number_of_lines=slice(cl_lo, cl_hi),
                pixels_per_line=slice(cp_lo, cp_hi)).values
            d2 = (lat_win - lat) ** 2 + (lon_win - lon) ** 2
            wi, wj = np.unravel_index(np.argmin(d2), d2.shape)
            cl, cp = cl_lo + wi, cp_lo + wj

            # Step 4: 5x5 box around the true nearest pixel
            ls = max(cl - 2, 0); le = min(cl + 3, nl_full)
            ps = max(cp - 2, 0); pe = min(cp + 3, np_full)
            v = ds_geo[variable_wanted].isel(
                number_of_lines=slice(ls, le),
                pixels_per_line=slice(ps, pe)).values
            f = ds_geo['l2_flags'].isel(
                number_of_lines=slice(ls, le),
                pixels_per_line=slice(ps, pe)).values

            valid = np.bitwise_and(f, exclude_mask) == 0

            if valid.any():
                v_valid = v[valid]
                mean0 = np.nanmean(v_valid, axis=0)
                std0 = np.nanstd(v_valid, axis=0)
                sel = np.all(np.abs(v_valid - mean0) <= 1.5 * std0, axis=1)
                mean = np.nanmean(v_valid[sel], axis=0).flatten()
                cv = np.nanstd(v_valid[sel], axis=0) / mean
                cv_med = np.nanmedian(
                    cv[(rrs_wavelengths >= 405) & (rrs_wavelengths <= 570)])
            else:
                cv_med = np.nan
                mean = np.full_like(rrs_wavelengths, np.nan, dtype=float)

            # Get the exact pixel coordinates (cheap — single value from the window)
            row = {
                'oci_datetime': granule_time,
                'oci_cv': cv_med,
                'oci_latitude': float(lat_win[wi, wj]),
                'oci_longitude': float(lon_win[wi, wj]),
                'oci_pixel_valid': int(valid.sum()),
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

            if valid.any():
                v_valid = v[valid]
                mean0 = np.nanmean(v_valid, axis=0)
                std0 = np.nanstd(v_valid, axis=0)
                sel = np.all(np.abs(v_valid - mean0) <= 1.5 * std0, axis=1)
                mean = np.nanmean(v_valid[sel], axis=0).flatten()
                cv = np.nanstd(v_valid[sel], axis=0) / mean
                cv_med = np.nanmedian(
                    cv[(rrs_wavelengths >= 405) & (rrs_wavelengths <= 570)])
            else:
                cv_med = np.nan
                mean = np.full_like(rrs_wavelengths, np.nan, dtype=float)

            row = {
                'oci_datetime': granule_time,
                'oci_cv': cv_med,
                'oci_latitude': sat_lat[cl, cp],
                'oci_longitude': sat_lon[cl, cp],
                'oci_pixel_valid': int(valid.sum()),
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

def match_data(df_sat, df_aoc, cv_max=0.15, senz_max=60.0,
               min_percent_valid=55.0, max_time_diff=180, std_max=1.5):
    """Create matchup dataframe based on selection criteria.

    Parameters
    ----------
    df_sat : pandas dataframe
        Satellite data from flat validation file.
    df_aoc : pandas dataframe
        Field data from flat validation file.
    cv_max : float, default 0.15
        Maximum coefficient of variation (stdev/mean) for sat data.
    senz_max : float, default 60.0
        Maximum sensor zenith for sat data.
    min_percent_valid : float, default 55.0
        Minimum percentage of valid satellite pixels.
    max_time_diff : int, default 180
        Maximum time difference (minutes) between sat and field matchup.
    std_max : float, default 1.5
        If multiple valid field matchups, select within std_max stdevs of mean.

    Returns
    -------
    pandas dataframe of matchups for product
    """
    # Setup
    time_window = pd.Timedelta(minutes=max_time_diff)
    df_match_list = []

    # Ensure both datetime columns are timezone-naive
    df_aoc['date'] = pd.to_datetime(df_aoc['date']).dt.tz_localize(None)
    df_sat['oci_datetime'] = pd.to_datetime(df_sat['oci_datetime']).dt.tz_localize(None)

    # Filter Field data based on Solar Zenith
    df_aoc_filtered = df_aoc

    # Filter satellite data based on cv threshold
    #df_sat_filtered = df_sat[df_sat['oci_cv'] <= cv_max]
    df_sat_filtered = df_sat
    df_sat_filtered = df_sat_filtered[
        df_sat_filtered['oci_pixel_valid'] >= min_percent_valid * 25 / 100]

    for _, sat_row in df_sat_filtered.iterrows():
        # Filter field data based on time difference and coordinates
        oci_datetime = pd.to_datetime(sat_row['oci_datetime'], errors='coerce').tz_localize(None)
        time_diff = abs(df_aoc_filtered['date'] - oci_datetime)
        within_time = time_diff <= time_window
        within_lat = 0.2 >= abs(df_aoc_filtered['lat'] - sat_row['oci_latitude'])
        within_lon = 0.2 >= abs(df_aoc_filtered['lon'] - sat_row['oci_longitude'])
        field_matches = df_aoc_filtered[within_time & within_lat & within_lon]

        if not field_matches.empty:
            # Select the best match based on time delta
            time_diff = abs(field_matches['date'] - oci_datetime)
            best_match = field_matches.loc[time_diff.idxmin()]
            df_match_list.append({**best_match.to_dict(), **sat_row.to_dict()})
            # Add a time_diff column
            df_match_list[-1]['time_diff'] = time_diff.min()

    df_match = pd.DataFrame(df_match_list)
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
kd_loc = kd_loc[kd_loc['quality'] != 2]


# REtrieve the Satellite data
#sat_cb = process_satellite_kd(kd_loc, sat="PACE KD")

sat_rrs = process_satellite_rrs_fast(kd_loc, sat="PACE IOP", variable_wanted='Kd',
    max_workers=4, download_dir='/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/pace_cache',
    checkpoint_path='/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_checkpoint.csv')

sat_rrs.to_csv('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/sat_kd_final.csv')

# Compute matchups
#matchups = match_data_kd(sat_cb, kd_loc)
matchups = match_data(sat_rrs, kd_loc,cv_max=0.8, senz_max=70.0,
               min_percent_valid=1.0, max_time_diff=380, std_max=6)
# Remove columns filled with NaN
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

# Disable interactive mode
plt.ioff()

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

# Define colors and symbols
colors = plt.cm.viridis(np.linspace(0, 1, len(matchups_clean)))

# Disable interactive mode
plt.ion()
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

#print first few lines of matchups
print(matchups.head())

#%% Make a nice map of the location of the matchups for slides

# Load Natural Earth shapefile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Load Natural Earth shapefile for countries
world = gpd.read_file('/Users/charlotte.begouen/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Read in matchups data (assuming matchups is defined elsewhere)
data = pd.DataFrame({
    'lon': matchups['lon'],  # Longitudes
    'lat': matchups['lat'],
    'oci_kd442': matchups['oci_kd442']  # Ensure this column exists# Latitudes
})
argo_loc = points = gpd.GeoDataFrame(kd_loc, geometry=gpd.points_from_xy(kd_loc.lon, kd_loc.lat), crs="EPSG:4326")

points = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs="EPSG:4326")
green_points = points[~points['oci_kd442'].isna()]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Base map (countries)
world.plot(ax=ax, color="lightgray", edgecolor="darkgrey")

# Plot points on top
argo_loc.plot(ax=ax, color="red", markersize=10,label='Hyperspectral profiles passing QC')
green_points.plot(ax=ax, color="green", markersize=10, label = 'Matchups with L2 Kd from PACE')

ax.legend(fontsize = 14)

# Set axis limits
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude",fontsize =16)
ax.set_ylabel("Latitude",fontsize=16)
plt.savefig('/Users/charlotte.begouen/Documents/PVST_Hyperspectral_floats_Herve/New_Outputs_NotRaw/Map_Location_Profiles.png')
plt.show()

#%% Plot of the comparison of PACE Kd vs float Kd

import matplotlib.pyplot as plt
import numpy as np
import re

# Extract wavelengths and corresponding values
wavelengths = []
kd_values = []
oci_kd_values = []

for _, row in matchups_clean.iterrows():
    for col in row.index:
        match_kd = re.match(r'kd(\d+)\.0$', col)
        match_oci_kd = re.match(r'oci_kd(\d+)$', col)
        if match_kd:
            wavelength = int(match_kd.group(1))
            wavelengths.append(wavelength)
            kd_values.append(row[col])
            oci_kd_values.append(row[f'oci_kd{wavelength}'])

# Convert to numpy arrays for plotting
wavelengths = np.array(wavelengths)
kd_values = np.array(kd_values)
oci_kd_values = np.array(oci_kd_values)

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(kd_values, oci_kd_values, c=wavelengths, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label='Wavelength (nm)')
plt.xlabel('Float Kd (kdXXX.0)', fontsize=14)
plt.ylabel('Satellite Kd (oci_kdXXX)', fontsize=14)
plt.title('Comparison of Float Kd vs Satellite Kd', fontsize=16)
plt.grid(True)
plt.show()