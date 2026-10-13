import warnings

import pandas as pd
from pyproj import Geod
from pysolar.solar import get_altitude

# pysolar lacks leap-second data for recent years; the error is ~1 s, negligible for SZA.
warnings.filterwarnings("ignore", message="Leap seconds for year", category=UserWarning)

geod = Geod(ellps="WGS84")


def match(sat, targets, cv_max=0.15, min_valid_pixels=10,
          time_window_minutes=180, max_distance_km=5, sza_max=75.0):
    """
    Match satellite and in-situ data following Bailey & Werdell (2006).
        - |time diff| < 180 min
        - number good pixels (NGP) > number total pixels (NTP) / 2 + 1
        - select pixel with NGP within +/-1.5 * RMS of median => median[CV] <= 0.15
        - solar zenith angle < 75 degrees
        - sensor zenith angle < 60 degrees

    Criteria applied:
      1. CV(405-570 nm) <= cv_max  (scene homogeneity) NOT FOR KD
      2. oci_pixel_valid >= min_valid_pixels  (>= 10 of 25 pixels)
      3. |time difference| <= max_time_diff minutes
      4. Solar zenith angle <= sza_max degrees  (daytime only)
    """
    sat = sat.copy()
    sat['oci_datetime'] = pd.to_datetime(sat['oci_datetime']).dt.tz_localize(None)

    time_window = pd.Timedelta(minutes=time_window_minutes)

    # ── Satellite-side filters ──────────────────────────────────────────────
    n0 = len(sat)

    # 1. Minimum valid pixels
    df_sat_f = sat[sat['oci_pixel_valid'] >= min_valid_pixels]
    print(f"  Min pixels (≥{min_valid_pixels}):      {n0} → {len(df_sat_f)} sat rows")

    # 2. CV threshold
    n1 = len(df_sat_f)
    df_sat_f = df_sat_f[df_sat_f['oci_cv'] <= cv_max]
    print(f"  CV filter (≤{cv_max}):    {n1} → {len(df_sat_f)} sat rows")

    # ── Build records: one row per (satellite row, its target) ─────────────
    records = df_sat_f.merge(targets, left_on='target_idx', right_index=True,
                             how='inner', suffixes=('_sat', ''))
    print(f"  Records (sat × target):   {len(records)}")

    # ── In-Sity Solar Zenith Angle filter ─────────────────────────────────────────
    n1 = len(records)
    records['sza'] = records.apply(compute_sza, axis=1)
    records = records[records['sza'] <= sza_max]
    print(f"  SZA filter  (≤{sza_max}°): {n1} → {len(records)} target rows")

    # ── Time filter ────────────────────────────────────────────────────────
    records['time_diff'] = (records['oci_datetime'] - records['datetime']).abs()
    n1 = len(records)
    records = records[records['time_diff'] <= time_window]
    print(f"  Time filter (≤{time_window_minutes} min): {n1} → {len(records)} records")

    # ── Distance filter (geodesic, metres → km) ────────────────────────────
    _, _, dist_m = geod.inv(records['lon'].values, records['lat'].values,
                            records['oci_longitude'].values, records['oci_latitude'].values)
    records['distance_km'] = dist_m / 1000.0
    n1 = len(records)
    records = records[records['distance_km'] <= max_distance_km]
    print(f"  Distance filter (≤{max_distance_km} km): {n1} → {len(records)} records")

    # ── Replicates: targets matched by more than one satellite row ─────────
    n1 = len(records['target_idx'])
    print(f"  Unique Targets (no satellite replicates): {n1} → {records['target_idx'].nunique()} records")

    # Keep the closest in time for each target
    records = (records.sort_values('time_diff')
               .drop_duplicates('target_idx', keep='first')
               .sort_index())
    print(f"Total Matchups: {len(records)}")
    return records


def compute_sza(row):
    if pd.isna(row['lat']) or pd.isna(row['lon']) or pd.isna(row['datetime']):
        return 90.0
    timestamp_utc = row['datetime'].tz_localize('UTC').to_pydatetime()
    return 90.0 - get_altitude(row['lat'], row['lon'], timestamp_utc)
