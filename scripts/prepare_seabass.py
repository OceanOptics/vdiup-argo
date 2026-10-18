"""Prepare the next SeaBASS submission.

1. Lists profiles already submitted (seabass/<REV>/submitted, searched recursively).
2. Copies processed Kd .sb files that contain not-yet-submitted profiles to
   seabass/<REV>/working/<wmo>/ (same file name replaces the previous
   submission on SeaBASS, so the whole monthly file is copied).
3. Prints/saves statistics (good/questionable/bad per float and month).
4. Reports anomalies relevant to a SeaBASS submission.
"""
import argparse
import glob
import os
import re
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd

from vdiup_argo.data_io.seabass import read_profiles
from vdiup_argo.config import PATH_TO_PROCESSED, PATH_TO_SEABASS, PATH_TO_SOURCE, QUALITY, SEABASS_REVISION

STALE_DAYS = 60          # float flagged if its last processed profile is older
MAX_CYCLE_GAP = 1        # report missing cycle numbers (gap > this)


def load_files(files):
    frames, problems = [], []
    for f in files:
        try:
            header, df = read_profiles(f)
        except Exception as e:  # noqa: BLE001
            problems.append(f'{f}: cannot be parsed ({e})')
            continue
        m = re.search(r'Kd_(\d+)_', os.path.basename(f))
        if m and m.group(1) != header.get('platform_id'):
            problems.append(f'{os.path.basename(f)}: platform_id {header.get("platform_id")} != WMO in file name')
        df['file'] = f
        frames.append(df)
    cols = ['wmo', 'profile', 'datetime', 'lat', 'lon', 'quality', 'kd_all_missing', 'file']
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)), problems


def stats_table(df):
    t = (df.assign(month=df['datetime'].dt.strftime('%Y-%m'),
                   q=df['quality'].map(QUALITY))
           .groupby(['wmo', 'month', 'q']).size().unstack('q', fill_value=0))
    for q in QUALITY.values():
        if q not in t:
            t[q] = 0
    t = t[list(QUALITY.values())]
    t['total'] = t.sum(axis=1)
    return t.reset_index()


def has_aux(source, wmo, cycle):
    return bool(glob.glob(os.path.join(source, wmo, 'profiles', f'*_{cycle:03d}*_aux.nc')))


def fmt_ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(nums) + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            out.append(f'{start}' if start == prev else f'{start}-{prev}')
            start = prev = n
    return ', '.join(out)


def main(revision, processed, root, dry_run, source=str(PATH_TO_SOURCE), skip_current_month=True):
    submitted_dir = os.path.join(root, revision, 'submitted')
    working_dir = os.path.join(root, revision, 'working')
    anomalies = []

    # --- 1. Submitted profiles
    sub_files = sorted(glob.glob(os.path.join(submitted_dir, '**', '*-Kd_*.sb'), recursive=True))
    sub, problems = load_files(sub_files)
    anomalies += [f'[submitted] {p}' for p in problems]
    print(f'Submitted ({revision}): {sub.wmo.nunique()} floats, {len(sub_files)} files, {len(sub)} profiles')
    if not os.path.isdir(submitted_dir):
        anomalies.append(f'[submitted] folder {submitted_dir} does not exist: treating everything as new')
    dup = sub[sub.duplicated(['wmo', 'profile'], keep=False)]
    for (w, p), g in dup.groupby(['wmo', 'profile']):
        anomalies.append(f'[submitted] {w} profile {p:03d} appears in {len(g)} submitted files '
                         f'({", ".join(os.path.basename(x) for x in g.file)})')
    sub = sub.drop_duplicates(['wmo', 'profile'], keep='last')

    # --- 2. Processed profiles
    proc_files = sorted(glob.glob(os.path.join(processed, '*', f'*-Kd_*_{revision}.sb')))
    proc, problems = load_files(proc_files)
    anomalies += [f'[processed] {p}' for p in problems]
    if proc.empty:
        raise SystemExit(f'No processed {revision} Kd files found in {processed}')
    dup = proc[proc.duplicated(['wmo', 'profile'], keep=False)]
    for (w, p), g in dup.groupby(['wmo', 'profile']):
        anomalies.append(f'[processed] {w} profile {p:03d} duplicated in {len(g)} files')
    proc = proc.drop_duplicates(['wmo', 'profile'], keep='last')

    def keys(d):
        return set(zip(d.wmo, d.profile))

    sub_keys = keys(sub)
    proc['new'] = [k not in sub_keys for k in zip(proc.wmo, proc.profile)]
    new = proc[proc.new]
    print(f'Processed: {proc.wmo.nunique()} floats, {len(proc_files)} files, {len(proc)} profiles [{len(new)} new in {new.file.nunique()} new files]')

    # --- 3. Copy files that contain new profiles
    files_to_copy = sorted(new.file.unique())
    skipped_current_month = []
    if skip_current_month:
        current_month = datetime.now().strftime('%Y-%m')
        current_month_files = set(new.loc[new.datetime.dt.strftime('%Y-%m') == current_month, 'file'])
        skipped_current_month = sorted(current_month_files)
        files_to_copy = [f for f in files_to_copy if f not in current_month_files]
    for f in files_to_copy:
        wmo = os.path.basename(os.path.dirname(f))
        dst_dir = os.path.join(working_dir, wmo)
        dst = os.path.join(dst_dir, os.path.basename(f))
        if os.path.exists(dst):
            anomalies.append(f'[working] {os.path.relpath(dst, root)} already existed and was overwritten')
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(f, dst)
        g = proc[proc.file == f]
        if (~g.new).any():
            anomalies.append(f'[working] {os.path.basename(f)}: {int((~g.new).sum())} profile(s) already submitted '
                             f'+ {int(g.new.sum())} new; file replaces the submitted one (same name)')
    print(f'{"Would copy" if dry_run else "Copied"} {len(files_to_copy)} file(s) to {working_dir}')
    if skipped_current_month:
        print(f'Skipped {len(skipped_current_month)} current-month file(s) because they may be incomplete')

    # --- 4. Statistics
    pd.set_option('display.width', 200, 'display.max_rows', 1000)
    st_new = stats_table(new) if len(new) else pd.DataFrame()
    st_all = stats_table(proc)
    cols = list(QUALITY.values()) + ['total']
    today = datetime.now().strftime('%Y-%m-%d')
    total = st_all.pivot(index='wmo', columns='month', values='total')
    newc = (st_new.pivot(index='wmo', columns='month', values='total')
            .reindex(index=total.index, columns=total.columns) if len(st_new) else total * 0).fillna(0)

    def cell(t, n):
        if pd.isna(t):
            return ''
        return f'{int(t)} ({int(n)})' if n > 0 else f'{int(t)}'

    table = pd.DataFrame({m: [cell(total.at[w, m], newc.at[w, m]) for w in total.index] for m in total.columns},
                         index=total.index)
    table.index.name = 'wmo'
    print('\n=== Profiles per float / month: total (new) ===')
    print(table.to_string())
    print('\n=== All processed profiles per float (total) ===')
    tot = st_all.groupby('wmo')[cols].sum()
    tot['new'] = new.groupby('wmo').size().reindex(tot.index).fillna(0).astype(int)
    print(tot.to_string())
    print('\n=== Overall per month (all floats, new profiles) ===')
    if len(st_new):
        print(st_new.groupby('month')[cols].sum().to_string())

    # --- 5. Anomalies
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    known_floats, proc_floats = set(sub.wmo), set(proc.wmo)
    for w in sorted(proc_floats - known_floats):
        anomalies.append(f'[float] {w}: NEW float, never submitted ({int((proc.wmo == w).sum())} profiles)')
    for w in sorted(known_floats - proc_floats):
        anomalies.append(f'[float] {w}: submitted before but absent from processed')
    for w in sorted(proc_floats & known_floats):
        if not (new.wmo == w).any():
            anomalies.append(f'[float] {w}: no new profiles since last submission')
    for w, g in proc.groupby('wmo'):
        if (g.quality == 2).all():
            anomalies.append(f'[quality] {w}: ALL {len(g)} profiles are bad')
        gn = g[g.new]
        if len(gn) and (gn.quality == 2).all():
            anomalies.append(f'[quality] {w}: all {len(gn)} NEW profiles are bad')
        last = g.datetime.max()
        if pd.notna(last) and now - last > timedelta(days=STALE_DAYS):
            anomalies.append(f'[float] {w}: last profile {last:%Y-%m-%d} is >{STALE_DAYS} days old')
        cyc = sorted(g.profile)
        missing = [c for a, b in zip(cyc, cyc[1:]) if b - a > MAX_CYCLE_GAP for c in range(a + 1, b)]
        if missing:
            with_aux = [c for c in missing if has_aux(source, w, c)]
            without_aux = [c for c in missing if c not in with_aux]
            msg = f'[cycles] {w}: {len(missing)} missing cycle(s)'
            if with_aux:
                msg += f'; _aux present in source but not processed: {fmt_ranges(with_aux)}'
            if without_aux:
                msg += f'; _aux absent from source: {fmt_ranges(without_aux)}'
            anomalies.append(msg)
        s = sub[sub.wmo == w]
        if len(s) and len(gn) and gn.datetime.min() < s.datetime.max():
            anomalies.append(f'[cycles] {w}: new profiles dated before last submitted profile '
                             f'({gn.datetime.min():%Y-%m-%d} < {s.datetime.max():%Y-%m-%d}), back-filled')
    for _, r in new[new.kd_all_missing & (new.quality != 2)].iterrows():
        anomalies.append(f'[data] {r.wmo} profile {r.profile:03d}: all Kd missing but quality={r.quality} (not bad)')
    for _, r in new.iterrows():
        if pd.isna(r.datetime):
            anomalies.append(f'[data] {r.wmo} profile {r.profile:03d}: invalid date/time')
        elif r.datetime > now:
            anomalies.append(f'[data] {r.wmo} profile {r.profile:03d}: date in the future ({r.datetime})')
        if pd.isna(r.lat) or pd.isna(r.lon) or abs(r.lat) > 90 or abs(r.lon) > 180:
            anomalies.append(f'[data] {r.wmo} profile {r.profile:03d}: invalid position ({r.lat}, {r.lon})')
        if pd.isna(r.quality) or r.quality not in QUALITY:
            anomalies.append(f'[data] {r.wmo} profile {r.profile:03d}: invalid quality flag {r.quality}')
    both = sub.merge(proc, on=['wmo', 'profile'], suffixes=('_sub', '_proc'))
    for _, r in both.iterrows():
        if r.quality_sub != r.quality_proc:
            anomalies.append(f'[changed] {r.wmo} profile {r.profile:03d}: quality {r.quality_sub} -> {r.quality_proc} since submission')
        if pd.notna(r.datetime_sub) and r.datetime_sub != r.datetime_proc:
            anomalies.append(f'[changed] {r.wmo} profile {r.profile:03d}: time {r.datetime_sub} -> {r.datetime_proc}')
        if abs(r.lat_sub - r.lat_proc) > 1e-4 or abs(r.lon_sub - r.lon_proc) > 1e-4:
            anomalies.append(f'[changed] {r.wmo} profile {r.profile:03d}: position changed since submission')
    for w, p in sorted(sub_keys - keys(proc)):
        anomalies.append(f'[missing] {w} profile {p:03d}: submitted but no longer in processed')

    os.makedirs(working_dir, exist_ok=True)
    table_file = os.path.join(working_dir, f'profiles_per_month_{today}.csv')
    table.to_csv(table_file)
    print(f'\nTable written to {table_file}')

    print(f'\n=== Anomalies ({len(anomalies)}) ===')
    print('\n'.join(sorted(anomalies)) if anomalies else 'none')
    anomalies_file = os.path.join(working_dir, f'anomalies_{today}.txt')
    with open(anomalies_file, 'w') as f:
        f.write('\n'.join(sorted(anomalies)) + '\n')
    print(f'Anomalies written to {anomalies_file}')
    print('\nReminder: run scripts/check_seabass.py (fcheck) before uploading.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-r', '--revision', default=SEABASS_REVISION)
    p.add_argument('--processed', default=str(PATH_TO_PROCESSED))
    p.add_argument('--source', default=str(PATH_TO_SOURCE))
    p.add_argument('--seabass', default=str(PATH_TO_SEABASS))
    p.add_argument('--dry-run', action='store_true', help='do not copy .sb files (reports are still written)')
    p.add_argument('--include-current-month', action='store_false', dest='skip_current_month',
                   help='copy current-month files (may be incomplete)')
    a = p.parse_args()
    main(a.revision, a.processed, a.seabass, a.dry_run, a.source, a.skip_current_month)
