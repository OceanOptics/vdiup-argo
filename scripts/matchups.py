import pandas as pd

from config import PATH_TO_PROCESSED, PATH_TO_MATCHUPS
from src.data_io import processed
from src.data_io.pace_oci import extract_pace_matchups
from src.process.matchups import match


def main(path_to_target_data=PATH_TO_PROCESSED, path_to_satellite_data=PATH_TO_MATCHUPS, variable='Kd',
         clear_cache=False, max_workers=None):
    # Load the target dataset
    target = processed.read(variable, path_to_target_data)
    print(f"Target dataset: {len(target)} rows loaded from {path_to_target_data} for variable '{variable}'")
    # Filter the target dataset
    n0 = len(target)
    target = target[target['quality'] != 'bad']
    print(f"  {variable.capitalize()} Quality (not bad): {n0} → {len(target)} targets")
    n0 = len(target)
    target = target.dropna(subset=['lat', 'lon', 'datetime'])
    print(f"  NaN date or position: {n0} → {len(target)} targets")
    n0 = len(target)
    PACE_OCI_FIRST_IMAGE_DATE = pd.to_datetime('2024-02-25')
    target = target[target['datetime'] >= PACE_OCI_FIRST_IMAGE_DATE]
    print(f"  Date >= {PACE_OCI_FIRST_IMAGE_DATE.date()}:   {n0} → {len(target)} targets")
    target = target.copy()  # Avoid performance warning due to filtering down the road
    # Download/Load PACE-OCI matchup data
    time_window_minutes = 180
    sat, granule_stats = extract_pace_matchups(target, sat="PACE IOP", variable_wanted=variable,
                                 max_workers=max_workers, time_window_minutes=time_window_minutes,
                                 download_dir=path_to_satellite_data / 'pace_granule_cache',
                                 clear_cache_first=clear_cache)
    # Refine matchups to only those with valid satellite data
    df = match(sat, target, cv_max=0.15, min_valid_pixels=10, time_window_minutes=time_window_minutes, sza_max=75.0)
    return df, granule_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Matchup target dataset with PACE-OCI.')
    parser.add_argument('path_to_target_dataset', type=str, default=PATH_TO_PROCESSED, nargs='?',
                        help='Path to target dataset (e.g. BGC-Argo Kd)')
    parser.add_argument('path_to_satellite_data', type=str, default=PATH_TO_MATCHUPS, nargs='?',
                        help='Path to satellite matchup data (default: PATH_TO_MATCHUPS)')
    parser.add_argument('--variable', type=str, default='Kd', choices=['Kd', 'Ed0', 'Ed'],
                        help='Variable to matchup (default: Kd)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete cached granules, search cache and checkpoint, then redownload everything')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of workers for parallel processing (default: None, which uses all available cores)')
    args = parser.parse_args()

    df, granule_stats = main(args.path_to_target_dataset, args.path_to_satellite_data, variable=args.variable,
              clear_cache=args.clear_cache, max_workers=args.max_workers)

    df.to_csv(args.path_to_satellite_data / f'PACE_OCI_{args.variable.upper()}.matchups.csv', index=False)
