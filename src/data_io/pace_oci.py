import concurrent.futures
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle

import earthaccess
import numpy as np
import pandas as pd
from pyproj import Geod
import xarray as xr
from tqdm import tqdm


geod = Geod(ellps="WGS84")


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
                   "ATMFAIL", "LOWLW", "FILTER", "NAVFAIL", "NAVWARN",]
                   # "HISOLZEN", "CHLFAIL"] # Added by NH

# Processing parameters (part of the checkpoint fingerprint)
MAX_PIXEL_DISTANCE_KM = 5.0  # max distance from target to closest pixel
HALF_WINDOW = 2          # 5x5 pixel window around the closest pixel
SIGMA_CLIP = 1.5         # pixels further than SIGMA_CLIP * std from the mean are dropped
# CV_BAND_NM = (405, 570)  # wavelength range used for the coefficient of variation (CBD values)
CV_BAND_NM = (412, 555)  # wavelength range used for the coefficient of variation (Values in Bailey & Werdell 2006)

SEARCH_CACHE_NAME = "_search_cache.pkl"
PROCESS_CHECKPOINT_NAME = "_process_checkpoint.csv"


class GranuleOpenError(Exception):
    """The local granule file could not be opened (likely corrupt)."""


def _atomic_write(path, writer, mode="w"):
    tmp = Path(f"{path}.tmp")
    with open(tmp, mode) as f:
        writer(f)
    os.replace(tmp, path)


def search_granules(targets, sat, time_window_minutes=180, cache_path=None):
    """Search once per unique (date and position). Returns [(profile_idx, granule), ...].

    Results are persisted in cache_path (keyed by satellite, date and position) so
    a re-run, even with different targets, only searches for new keys. Failed searches are
    not cached and are retried next time.
    """
    short_name = SAT_LOOKUP[sat]
    time_window = pd.Timedelta(minutes=time_window_minutes)

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"  ignoring unreadable search cache ({e})")
    n_initial = len(cache)


    failed = {}
    pairs = []
    try:
        for idx, row in tqdm(targets.iterrows(), total=len(targets), desc="Searching granules"):
            ckey = (short_name, row['datetime'].isoformat(), row['lat'], row['lon'])
            if ckey not in cache and ckey not in failed:
                try:
                    cache[ckey] = earthaccess.search_data(
                        temporal=(
                            (row['datetime'] - time_window).strftime('%Y-%m-%dT%H:%M:%SZ'),
                            (row['datetime'] + time_window).strftime('%Y-%m-%dT%H:%M:%SZ')
                        ),
                        point=(row['lon'], row['lat']),
                        short_name=short_name)
                except Exception as e:
                    tqdm.write(f"  search failed for {ckey}: {e}")
                    failed[ckey] = []
            pairs.extend((idx, granule) for granule in cache.get(ckey, failed.get(ckey, [])))
    finally:
        if cache_path and len(cache) > n_initial:
            _atomic_write(cache_path, lambda f: pickle.dump(cache, f), mode="wb")
    return pairs


def _expected_path(granule, download_dir):
    """Local path earthaccess uses for a granule (last URL segment of its .nc link)."""
    links = [u for u in granule.data_links() if u.endswith(".nc")]
    return Path(download_dir) / links[0].split("/")[-1]


def download_granules(granules, download_dir, threads=8):
    """Download missing granules. Returns {gid: local_path} for granules present on disk.

    earthaccess writes to a temporary sibling file and renames it on completion, so a file at
    its final path is complete; interrupted downloads are simply redone on the next run.
    """
    paths = {gid: _expected_path(g, download_dir) for gid, g in granules.items()}
    missing = [gid for gid, path in paths.items() if not path.exists()]
    print(f"  {len(paths) - len(missing)} granules cached, {len(missing)} to download")
    if missing:
        earthaccess.login()
        results = earthaccess.download(
            [granules[g] for g in missing], download_dir, threads=threads,
            show_progress=True,
            pqdm_kwargs={"exception_behaviour": "deferred", "desc": "Downloading granules"})
        n_failed = sum(isinstance(r, Exception) for r in results)
        if n_failed:
            print(f"  {n_failed} downloads failed (will be retried on next run)")
    return {gid: path for gid, path in paths.items() if path.exists()}


def _fingerprint(targets, sat, variable_wanted):
    """Hash of everything that determines the processed results."""
    h = hashlib.sha256()
    h.update(repr((sat, variable_wanted, sorted(EXCLUSION_FLAGS), HALF_WINDOW,
                   SIGMA_CLIP, CV_BAND_NM)).encode())
    h.update(pd.util.hash_pandas_object(targets[['datetime', 'lat', 'lon']],
                                        index=True).values.tobytes())
    return h.hexdigest()


def _load_checkpoint(checkpoint_path, fingerprint):
    """Return (rows, done_gids); empty if missing or produced with different inputs."""
    meta_path = Path(f"{checkpoint_path}.meta.json")
    if not (checkpoint_path.exists() and meta_path.exists()):
        return [], set()
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        print("  checkpoint was made with different targets/parameters; ignoring it")
        return [], set()
    done = set(meta["done_gids"])
    ckpt = pd.read_csv(checkpoint_path)
    # Drop rows of granules not marked done (crash between CSV and meta writes)
    ckpt = ckpt[ckpt['_gid'].isin(done)]
    return ckpt.to_dict('records'), done


def _save_checkpoint(checkpoint_path, fingerprint, rows, done):
    columns = list(rows[0].keys()) if rows else ['_gid']
    _atomic_write(checkpoint_path,
                  lambda f: pd.DataFrame(rows, columns=columns).to_csv(f, index=False))
    _atomic_write(f"{checkpoint_path}.meta.json",
                  lambda f: json.dump({"fingerprint": fingerprint,
                                       "done_gids": sorted(done)}, f))


def clear_cache(download_dir):
    """Delete cached granules, the search cache and the checkpoint."""
    download_dir = Path(download_dir)
    files = list(download_dir.glob("*.nc")) + [
        download_dir / SEARCH_CACHE_NAME,
        download_dir / PROCESS_CHECKPOINT_NAME,
        download_dir / f"{PROCESS_CHECKPOINT_NAME}.meta.json"
    ]
    n = 0
    for f in files:
        if f.exists():
            f.unlink()
            n += 1
    print(f"  cleared cache: {n} files removed")


def extract_granule_stats(local_path, granule, target_specs,
                          rrs_wavelengths, variable_wanted):
    """Extract matchups for the given profiles from a local .nc granule.

    Raises GranuleOpenError if the file cannot be opened.
    """
    try:
        ds_nav = xr.open_dataset(local_path, group="navigation_data")
        ds_geo = xr.open_dataset(local_path, group="geophysical_data")
    except Exception as e:
        raise GranuleOpenError(f"{local_path}: {e}") from e

    granule_time = pd.to_datetime(
        granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    exclude_mask = sum(L2_FLAGS[f] for f in EXCLUSION_FLAGS)
    prefix = f'oci_{variable_wanted.lower()}'

    try:
        # Now that the file is local, just read full arrays — it's all memory.
        sat_lat = ds_nav['latitude'].values
        sat_lon = ds_nav['longitude'].values
        var_data = ds_geo[variable_wanted].values
        flags_data = ds_geo['l2_flags'].values

        cv_band_mask = (rrs_wavelengths >= CV_BAND_NM[0]) & (rrs_wavelengths <= CV_BAND_NM[1])

        out = []
        for target_idx, lat, lon in target_specs:
            distances = (sat_lat - lat) ** 2 + (sat_lon - lon) ** 2
            cl, cp = np.unravel_index(np.argmin(distances), distances.shape)

            # Wrapped, cos(lat)-scaled difference: correct away from the equator and across ±180°
            dlon = (sat_lon - lon + 180.0) % 360.0 - 180.0
            d2 = (sat_lat - lat) ** 2 + (dlon * np.cos(np.radians(lat))) ** 2
            cl, cp = np.unravel_index(np.nanargmin(d2), d2.shape)

            _, _, dist_m = geod.inv(lon, lat, sat_lon[cl, cp], sat_lat[cl, cp])
            distance_km = dist_m / 1000.0
            if distance_km > MAX_PIXEL_DISTANCE_KM:
                continue  # target lies outside the swath: no statistics, no row

            ls = max(cl - HALF_WINDOW, 0); le = min(cl + HALF_WINDOW + 1, sat_lat.shape[0])
            ps = max(cp - HALF_WINDOW, 0); pe = min(cp + HALF_WINDOW + 1, sat_lat.shape[1])
            v = var_data[ls:le, ps:pe]
            f = flags_data[ls:le, ps:pe]

            # v is (≤5, ≤5, n_wl) for 3D products (IOP) or (≤5, ≤5) for 2D.
            # Normalise to 3D so the rest of the logic is identical.
            if v.ndim == 2:
                v = v[:, :, np.newaxis]
            # Keep only unflagged pixels with a finite value in every band.
            valid = (np.bitwise_and(f, exclude_mask) == 0) & np.isfinite(v).all(axis=2)
            if not valid.any():
                continue  # no valid pixels: no statistics, no row

            v_valid = v[valid, :]  # (n_valid_px, n_wl)
            mean0 = v_valid.mean(axis=0)
            std0 = v_valid.std(axis=0)
            sel = np.all(np.abs(v_valid - mean0) <= SIGMA_CLIP * std0, axis=1)
            v_final = v_valid[sel, :]  # never empty: not all pixels can exceed 1.5 sigma
            n_valid_final = len(v_final)
            mean = v_final.mean(axis=0)  # (n_wl,)
            cv = v_final.std(axis=0) / np.where(mean == 0, np.nan, mean)
            cv_band = cv[cv_band_mask]
            cv_med = np.nanmedian(cv_band) if np.isfinite(cv_band).any() else np.nan

            row = {
                'target_idx': target_idx,
                'oci_datetime': granule_time,
                'oci_cv': cv_med,
                'oci_latitude': sat_lat[cl, cp],
                'oci_longitude': sat_lon[cl, cp],
                'oci_pixel_valid': n_valid_final,  # now post-sigma filter
            }

            for wv, mv in zip(rrs_wavelengths, mean):
                row[f'{prefix}{int(wv)}'] = mv
            out.append(row)
    finally:
        ds_nav.close()
        ds_geo.close()

    return out

def extract_rrs_wavelengths(file):
    """Extract wavelengths — handles both full hyperspectral (AOP) and
    17-band IOP products automatically."""
    with xr.open_dataset(file, group="sensor_band_parameters") as ds:
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


def extract_pace_matchups(targets, variable_wanted, sat="PACE IOP", time_window_minutes=180,
                          max_workers=None, download_dir=None,
                          checkpoint_every=50, clear_cache_first=False):
    """Search, download all granules to disk in parallel, then process them locally.

    Everything is resumable (Ctrl+C safe):
      - download_dir caches the .nc files and the granule search results,
      - checkpoint_path (CSV + .meta.json) stores processed rows and the processed granule
        ids; it is ignored if the targets or processing parameters changed.
    clear_cache_first: delete cached granules, search cache and checkpoint, then start over.
    """
    if download_dir is None:
        download_dir = Path.cwd() / "pace_granule_cache"
    download_dir = Path(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    if clear_cache_first:
        clear_cache(download_dir)

    # ---- Phase 1: search granules ----
    pairs = search_granules(targets, sat, time_window_minutes, download_dir / SEARCH_CACHE_NAME)
    by_granule = defaultdict(list)
    granule_objs = {}
    for target_idx, granule in pairs:
        gid = granule["meta"]["concept-id"]
        by_granule[gid].append(target_idx)
        granule_objs[gid] = granule
    print(f"  {len(pairs)} profile-granule pairs → {len(by_granule)} unique granules")

    # ---- Resume: load checkpoint ----
    fingerprint = _fingerprint(targets, sat, variable_wanted)
    all_rows, done = [], set()
    checkpoint_path = Path(download_dir) / PROCESS_CHECKPOINT_NAME
    if checkpoint_path:
        all_rows, done = _load_checkpoint(checkpoint_path, fingerprint)
        if done:
            print(f"  resuming: {len(done)} granules already processed")
    todo_gids = [g for g in by_granule if g not in done]
    print(f"  {len(todo_gids)} granules to process")

    def _result():
        return pd.DataFrame(all_rows).drop(columns=['_gid'], errors='ignore')

    if not todo_gids:
        return _result()

    # ---- Phase 2: download only what is missing ----
    gid_to_path = download_granules({g: granule_objs[g] for g in todo_gids},
                                    download_dir, threads=8)
    todo_gids = [g for g in todo_gids if g in gid_to_path]
    if not todo_gids:
        return _result()

    # ---- Phase 3: process local files (no network) ----
    rrs_wavelengths = extract_rrs_wavelengths(gid_to_path[todo_gids[0]])

    def _job(gid):
        specs = [(idx, targets.loc[idx, 'lat'], targets.loc[idx, 'lon']) for idx in by_granule[gid]]
        try:
            rows = extract_granule_stats(gid_to_path[gid], granule_objs[gid], specs,
                                         rrs_wavelengths, variable_wanted)
        except GranuleOpenError:
            gid_to_path[gid].unlink(missing_ok=True)  # corrupt: re-download next run
            raise
        for r in rows:
            r['_gid'] = gid
        return rows

    n_error = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {ex.submit(_job, g): g for g in todo_gids}
    try:
        for n_done, fut in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                          total=len(futures), desc="Processing granules"), 1):
            try:
                all_rows.extend(fut.result())
                done.add(futures[fut])  # includes granules that yielded no rows
            except Exception as e:
                n_error += 1
                if n_error <= 5:
                    tqdm.write(f"  process failed for {futures[fut]}: {e}")
            if n_done % checkpoint_every == 0:
                _save_checkpoint(checkpoint_path, fingerprint, all_rows, done)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        ex.shutdown()
    finally:
        _save_checkpoint(checkpoint_path, fingerprint, all_rows, done)
    print(f"  done: {len(all_rows)} rows, {n_error} errors")
    return _result()
