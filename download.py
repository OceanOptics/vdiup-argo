import re
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter

ROOT = Path('/Volumes/SD/VDIUP/Argo/CBD')
BASE = 'https://data-argo.ifremer.fr'
WORKERS = 16
# Progress bars need a real terminal. In PyCharm, enable
# Run/Debug Configuration > "Emulate terminal in output console",
# or set USE_TQDM = False to fall back to plain text output.
USE_TQDM = True

session = requests.Session()
session.mount('https://', HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS))


def list_remote(url, pattern='*'):
    """Recursively yield (url, relative Path) of files below a directory URL."""
    response = session.get(url, timeout=60)
    response.raise_for_status()
    html = response.text
    for href in re.findall(r'href="([^"?#]+)"', html):
        if href.startswith(('/', '..', 'http')):
            continue
        if href.endswith('/'):
            for u, rel in list_remote(urljoin(url, href), pattern):
                yield u, Path(href.rstrip('/')) / rel
        elif fnmatch(href, pattern):
            yield urljoin(url, href), Path(href)


def sync_file(args):
    """Download unless local file has the same size and is not older than remote."""
    url, dest = args
    head = session.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    size = int(head.headers.get('Content-Length', -1))
    modified = parsedate_to_datetime(head.headers['Last-Modified']).timestamp() \
        if 'Last-Modified' in head.headers else None
    if dest.exists():
        stat = dest.stat()
        if stat.st_size == size and (modified is None or stat.st_mtime >= modified):
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return True


def sync_dir(url, dest_dir, pattern='*', desc=''):
    try:
        jobs = [(u, dest_dir / rel) for u, rel in list_remote(url, pattern)]
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            log(f"WARNING: {desc}: not found on server ({url}), check the DAC column")
            return
        raise
    if not jobs:
        log(f"WARNING: {desc}: no matching files at {url}")
        return
    with ThreadPoolExecutor(WORKERS) as pool:
        results = pool.map(sync_file, jobs)
        if USE_TQDM:
            results = tqdm(results, total=len(jobs), desc=desc, unit='file', leave=False)
        downloaded = sum(results)
    log(f"{desc}: {downloaded} downloaded, {len(jobs) - downloaded} up to date")


def log(message):
    if USE_TQDM:
        tqdm.write(message)
    else:
        print(message, flush=True)


floats = pd.read_csv(ROOT / 'HyperFloats.csv')
floats['DAC'] = floats['DAC'].fillna('coriolis').str.strip().str.lower().replace('', 'coriolis')
floats = floats.drop_duplicates('WMO')

pairs = list(zip(floats['WMO'], floats['DAC']))
progress = tqdm(pairs, desc='Floats', unit='float', dynamic_ncols=True) if USE_TQDM else pairs
for number, dac in progress:
    dir_path = ROOT / str(number)
    general_path = dir_path / 'profiles_general'
    general_path.mkdir(parents=True, exist_ok=True)

    # standard B-files: needed for TEMP/PSAL interpolated to Ed depths
    sync_dir(f'{BASE}/dac/{dac}/{number}/profiles/', general_path, 'R*.nc', f'{number} general')
    # aux: contains both raw counts and calibrated DOWN_IRRADIANCE_SPECTRUM
    sync_dir(f'{BASE}/aux/{dac}/{number}/', dir_path, desc=f'{number} aux')
