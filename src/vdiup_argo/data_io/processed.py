from pathlib import Path

import pandas as pd

from vdiup_argo.config import PATH_TO_PROCESSED, QUALITY

PRODUCTS = ('Kd', 'Ed0', 'Ed')


def read(product, path=PATH_TO_PROCESSED, recursive=True):
    """Read all {wmo}_{product}.csv files under `path` for one product (Kd, Ed0 or Ed).

    Returns a single DataFrame of the concatenated files with a 'wmo' column added,
    'date' and 'time' merged into a 'datetime' column, and 'quality' converted to
    the labels good / questionable / bad.
    """
    if product not in PRODUCTS:
        raise ValueError(f'product must be one of {PRODUCTS}, got {product!r}')
    path = Path(path)
    glob = path.rglob if recursive else path.glob
    suffix = f'_{product}.csv'
    frames, wmos = [], []
    for f in sorted(glob(f'*{suffix}')):
        if f.name.startswith('.'):  # macOS '._*' metadata files
            continue
        df = pd.read_csv(f, dtype={'date': str, 'time': str})
        wmos.append(f.name[:-len(suffix)])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No *{suffix} files found in {path}')
    df = pd.concat(frames, ignore_index=True)
    wmo = pd.Series(wmos).repeat([len(fr) for fr in frames]).reset_index(drop=True).rename('wmo')
    dt = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H:%M:%S', errors='coerce')
    df = pd.concat([wmo, dt.rename('datetime'), df.drop(columns=['date', 'time'])], axis=1)
    if 'quality' in df:
        df['quality'] = pd.Categorical(df['quality'].map(QUALITY), categories=list(QUALITY.values()))
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read all {wmo}_{product}.csv files under `path` for one product (Kd, Ed0 or Ed).')
    parser.add_argument('--product', type=str, default='Kd', choices=PRODUCTS, help='Product to read (Kd, Ed0 or Ed, default: Kd)')
    parser.add_argument('--path', type=str, default=PATH_TO_PROCESSED, help='Path to processed data (default: PATH_TO_PROCESSED)')
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=True, help='Search recursively in subdirectories (--no-recursive to disable, default: True)')
    args = parser.parse_args()

    df = read(args.product, path=args.path, recursive=args.recursive)
    print(df.head())
