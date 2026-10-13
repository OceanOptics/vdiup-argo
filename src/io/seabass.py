"""Read SeaBASS files written by seabass_maker.format_to_seabass."""
import os
import re
from io import StringIO

import pandas as pd

REQUIRED_FIELDS = ['profile', 'date', 'time', 'lon', 'lat', 'quality']
DELIMITERS = {'comma': ',', 'tab': '\t', 'space': ' '}


def read_sb(filename):
    """Read a Kd SeaBASS file.

    Returns (df, header). Columns of df are a MultiIndex (name, wavelength): kd, kd_unc, kd_se and
    kd_bincount have the wavelength as second level, other fields have ''. 'date' and 'time' are
    merged into a 'datetime' column (position 1). Missing values (-9999) are NaN.
    """
    # Header lines look like "/key=value" and end with "/end_header"; the rest is data
    header, data_lines, in_data = {}, [], False
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if in_data:
                if line.strip():
                    data_lines.append(line)
            elif line.startswith('/end_header'):
                in_data = True
            elif line.startswith('/') and '=' in line:
                key, value = line[1:].split('=', 1)
                header[key] = value
    fields = header['fields'].split(',')
    for f in REQUIRED_FIELDS:
        if f not in fields:
            raise KeyError(f"{os.path.basename(filename)}: missing field {f} in file.")
    delim = DELIMITERS.get(header.get('delimiter', 'comma'), ',')
    df = pd.read_csv(StringIO('\n'.join(data_lines)), names=fields, sep=delim, na_values=-9999,
                     dtype={'date': str, 'time': str})
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(
        [(f'kd{f[7:]}', float(f[2:7])) if f.startswith('kd') else (f, '') for f in fields])
    df.insert(1, ('datetime', ''),
              pd.to_datetime(df[('date', '')] + df[('time', '')], format='%Y%m%d%H:%M:%S', errors='coerce'))
    df = df.drop(columns=[('date', ''), ('time', '')])
    if 'kd' in df.columns.get_level_values(0):
        for e in ['_unc', '_se', '_bincount']:
            try:
                if not (df['kd'].columns == df[f'kd{e}'].columns).all():
                    raise KeyError(f"{os.path.basename(filename)}: missing wavelength kd{e} in file.")
            except ValueError as ex:
                missing = [c for c in df['kd'].columns if c not in df[f'kd{e}'].columns]
                raise ValueError(f"{os.path.basename(filename)}: {ex} kd and kd{e} columns. "
                                 f"Missing wavelength(s): {missing}")
    return df, header


def read_sb_profiles(filename):
    """Return (header, per-profile summary DataFrame) of a Kd SeaBASS file."""
    df, header = read_sb(filename)
    kd_missing = df['kd'].isna().all(axis=1) if 'kd' in df.columns.get_level_values(0) else True
    out = pd.DataFrame({
        'wmo': header.get('platform_id', ''),
        'profile': df[('profile', '')].astype(int),
        'datetime': df[('datetime', '')],
        'lat': df[('lat', '')],
        'lon': df[('lon', '')],
        'quality': df[('quality', '')].astype('Int64'),
        'kd_all_missing': kd_missing,
    })
    return header, out
