"""Merge all Kd profiles from SeaBASS files (Updated_KdCompute_Code output) into one CSV for report.py."""
import argparse
import glob
import os

import pandas as pd

from src.io.seabass import read_sb
from config import PATH_TO_PROCESSED, PATH_TO_MATCHUPS

DEFAULT_OUT = PATH_TO_MATCHUPS / 'All_Kd_profiles.csv'


def load_file(filename):
    df, header = read_sb(filename)
    wmo = header['platform_id']
    out = {
        'date': df[('datetime', '')],
        'lat': df[('lat', '')],
        'lon': df[('lon', '')],
        'WMO': [f"{wmo}_{int(p):03d}" for p in df[('profile', '')]],
        'quality': df[('quality', '')],
    }
    for var in ['kd', 'kd_unc']:
        suffix = '_unc' if var.endswith('_unc') else ''
        for wl in df[var].columns:
            out[f'kd{float(wl)}{suffix}'] = df[(var, wl)]
    return pd.DataFrame(out)


def main(dirname, output):
    files = sorted(glob.glob(os.path.join(dirname, '**', '*-Kd_*.sb'), recursive=True))
    if not files:
        raise SystemExit(f'No Kd SeaBASS files found in {dirname}')
    frames = [load_file(f) for f in files]
    a = pd.concat(frames, ignore_index=True).sort_values(['WMO']).drop_duplicates('WMO', keep='last')
    meta = ['date', 'lat', 'lon', 'WMO', 'quality']
    kd = sorted([c for c in a.columns if c.startswith('kd') and not c.endswith('_unc')], key=lambda c: float(c[2:]))
    a = a[meta + kd + [f'{c}_unc' for c in kd]]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    a.to_csv(output, index=False)
    print(f'{len(a)} profiles from {len(files)} files written to {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dirname', nargs='?', default=PATH_TO_PROCESSED, help='directory with SeaBASS Kd files (searched recursively)')
    p.add_argument('-o', '--output', default=DEFAULT_OUT)
    args = p.parse_args()
    main(args.dirname, args.output)
