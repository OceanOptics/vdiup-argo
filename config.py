from pathlib import Path

# SeaBASS submission revision (suffix of the .sb files, subfolder of PATH_TO_SEABASS)
SEABASS_REVISION = 'R2'

ROOT = Path('/Volumes/SD/VDIUP/Argo/')

PATH_TO_SOURCE = ROOT / 'source'
PATH_TO_PROCESSED = ROOT / 'processed'
PATH_TO_MATCHUPS = ROOT / 'matchups'
PATH_TO_FIGURES = ROOT / 'figures'
PATH_TO_SEABASS = ROOT / 'seabass'
PATH_TO_SEABASS_SUBMISSION = PATH_TO_SEABASS / SEABASS_REVISION / 'working'

FLOAT_LIST_FILE = ROOT / 'HyperFloats.csv'

DOWNLOAD_DATE = '20261009'

