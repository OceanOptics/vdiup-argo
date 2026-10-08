import glob
import os.path
import logging
import shutil

from tqdm import tqdm

from check.dcheck import make_plot
from check.fcheck import run_fcheck

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    target_revision = 'R2'
    # dirname = '/Users/nils/Documents/Lab/Projects/VDIUP/_Floats/submitted/20260622'
    dirname = '/Volumes/SD/VDIUP/Argo/CBD/New_Outputs_NotRaw'
    fig_dir = dirname + '_figs'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        if not os.path.exists(os.path.join(fig_dir, 'html')):
            os.makedirs(os.path.join(fig_dir, 'html'))
    # Run Format Check
    # for f in tqdm(sorted(glob.glob(os.path.join(dirname, '*.sb'))), desc='fcheck'):
    for f in tqdm(sorted(glob.glob(os.path.join(dirname, '**', '*.sb'))), desc='fcheck'):
        result = run_fcheck(f)
        if not result[0]:
            logger.error(f"{os.path.basename(f)}: Failed fcheck4.")
            for error in result[1]:
                logger.debug(error.decode().strip())
    # Run Data Check
    # platforms = {f.split('_')[2] for f in glob.glob('*.sb', root_dir=dirname)}  # all files in same dir
    platforms = os.listdir(dirname)  # one dir per float
    for p in tqdm(platforms, desc='dcheck'):
        try:
            # fig = make_plot(sorted(glob.glob(os.path.join(dirname, f'*_{p}_*.sb'))))
            fig = make_plot(sorted(glob.glob(os.path.join(dirname, p, f'*_{p}_*.sb'))))
            fig.write_image(os.path.join(fig_dir, f'{p}.png'), width=1920, height=1400)
            fig.write_html(os.path.join(fig_dir, 'html', f'{p}.html'))
        except (KeyError, ValueError) as e:
            logger.error(e)

    # Clean directories - keep only sb files.
    # other_dir = dirname + '_other'
    # if not os.path.exists(other_dir):
    #     os.makedirs(other_dir)
    # for p in tqdm(platforms, desc='clean'):
    #     for f in sorted(glob.glob(os.path.join(dirname, p, '*'))):
    #         if not f.endswith('.sb'):
    #             # move to other_dir
    #             if not os.path.exists(os.path.join(other_dir, p)):
    #                 os.makedirs(os.path.join(other_dir, p))
    #             shutil.move(f, os.path.join(other_dir, p, os.path.basename(f)))

    # Check revision numbers in .sb files and rename R3 to R2
    # for p in tqdm(platforms, desc='rcheck'):
    #     for f in sorted(glob.glob(os.path.join(dirname, p, '*.sb'))):
    #         basename = os.path.basename(f)
    #         # Check if filename contains revision marker
    #         if '_R' in basename:
    #             # Extract revision number (should be R2 or R3)
    #             parts = basename.split('_R')
    #             if len(parts) >= 2:
    #                 revision_part = parts[-1].split('.')[0]  # Get revision before .sb
    #                 revision = 'R' + revision_part
    #                 if revision != target_revision:
    #                     logger.warning(f"File {basename} has revision {revision}, expected R2")
    #                     # Only rename R3
    #                     if revision == 'R3':
    #                         new_basename = basename.replace('_R3.sb', f'_{target_revision}.sb')
    #                         new_path = os.path.join(dirname, p, new_basename)
    #                         os.rename(f, new_path)
    #                         logger.info(f"Renamed {basename} to {new_basename}")
