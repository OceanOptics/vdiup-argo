"""Read SeaBASS files written by seabass_maker.format_to_seabass."""
import os
import re
from io import StringIO

import pandas as pd

REQUIRED_FIELDS = ['profile', 'date', 'time', 'lon', 'lat', 'quality']
DELIMITERS = {'comma': ',', 'tab': '\t', 'space': ' '}


def read(filename):
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


def read_profiles(filename):
    """Return (header, per-profile summary DataFrame) of a Kd SeaBASS file."""
    df, header = read(filename)
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


def write(data, metadata, filename, path, comments=None, missing_value_placeholder='-9999', delimiter='comma'):
    """
    Formats the data and metadata into NASA SeaBASS format and writes to a .sb file.

    Parameters:
    - data (pd.DataFrame): The dataset to be formatted.
    - metadata (dict): A dictionary containing metadata information.
    - filename (str): The output file name without extension.
    - path (str): The path where the file will be saved.
    - comments (list, optional): A list of comments to include in the file.
    - missing_value_placeholder (str): The placeholder for missing values.
    - delimiter (str): The delimiter to use ('comma', 'tab', or 'space').
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.sb'):
        filename += '.sb'
    metadata['data_file_name'] = filename

    # Handle missing values in datetime columns separately
    datetime_cols = data.select_dtypes(include=[np.datetime64]).columns
    data[datetime_cols] = data[datetime_cols].fillna(pd.Timestamp('19000101'))

    # Replace missing values in other columns with '-9999'
    other_cols = data.columns.difference(datetime_cols)
    #data[other_cols] = data[other_cols].fillna(missing_value_placeholder).infer_objects(copy=False)



    # Ensure comments is a list
    if comments is None:
        comments = []
    elif isinstance(comments, str):
        comments = [comments]
    else:
        comments = comments.copy()  # Create a local copy of the comments list

    # Determine the delimiter
    if delimiter == 'comma':
        delim = ','
    elif delimiter == 'tab':
        delim = '\t'
    elif delimiter == 'space':
        delim = ' '
    else:
        raise ValueError("Unsupported delimiter. Choose from 'comma', 'tab', or 'space'.")

    if '_Ed' in filename:
        match = re.search(r'_(\d{3})_', filename)
        if match:
            cycle = match.group(1)
            metadata['profile'] = cycle
            metadata['id_fields_definitions'] = '1id:pre-tilt,2id:post-tilt'
            metadata['below_detection_limit'] = '-8888'
            comments.append(
                'tilt_1id= pre-tilt, i.e. tilt of the instrument just before performing performing a radiometric measurement')
            comments.append('tilt = post-tilt, i.e. tilt of the instrument just after performing a radiometric measurement.')
            metadata.pop('measurement_depth', None)

    elif '-Kd' in filename:
        metadata['measurement_depth'] = 0
        metadata.pop('profile', None)
        metadata.pop('below_detection_limit',None)

    comments.append("")

    # Write the data and metadata to a .sb file
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as f:
        # Write header information
        f.write('/begin_header\n')
        for key, value in metadata.items():
            if isinstance(value, list):
                f.write(f'/{key}={",".join(map(str, value))}\n')
            else:
                f.write(f'/{key}={value}\n')
        f.write(f'/missing={missing_value_placeholder}\n')
        f.write(f'/delimiter={delimiter}\n')

        # Write calculated metadata (dates, times, locations)
        start_date = data['date'].min()
        end_date = data['date'].max()
        start_time = data['time'].min()
        end_time = data['time'].max()
        north_latitude = data['lat'].max()
        south_latitude = data['lat'].min()
        east_longitude = data['lon'].max()
        west_longitude = data['lon'].min()

        f.write(f'/start_date={start_date}\n')
        f.write(f'/end_date={end_date}\n')
        f.write(f'/start_time={start_time}[GMT]\n')
        f.write(f'/end_time={end_time}[GMT]\n')
        f.write(f'/north_latitude={north_latitude}[DEG]\n')
        f.write(f'/south_latitude={south_latitude}[DEG]\n')
        f.write(f'/east_longitude={east_longitude}[DEG]\n')
        f.write(f'/west_longitude={west_longitude}[DEG]\n')

        if comments:
            f.write('!\n! COMMENTS\n')
            for comment in comments:
                f.write(f'! {comment}\n')

        if '_Ed' in filename:
            data = data.drop(columns=['lat', 'lon'])

        # Define units for each type of data
        units_dict = {
            'station': 'none',
            'date': 'yyyymmdd',
            'depth': 'm',
            'wt': 'degreesC',
            'sal': 'psu',
            'tilt': 'degrees',
            'time': 'hh:mm:ss',
            'lat': 'degrees',
            'lon': 'degrees',
            'quality': 'none',
            'kd': '1/m',
            'ed': 'uW/cm^2/nm',
            'Epar': 'uE/cm^2/s',
            'bincount' : 'none',
        }

        formatt = {
            'station': 's',
            'date': 's',
            'depth': '.4f',
            'wt': '.4f',
            'sal': '.4f',
            'tilt': '.1f',
            'time': 's',
            'lat': '.5f',
            'lon': '.5f',
            'quality': 'd',
            'kd': '.4f',
            'ed': '.4f',
            'Epar': '.4f',
            'profile': 'd',
        }

        units_list = []
        for col in data.columns:
            if 'kd' in col and 'bincount' in col:
                units_list.append('none')
            elif 'kd' in col:
                units_list.append('1/m')
            else:
                for key, unit in units_dict.items():
                    if key in col:
                        units_list.append(unit)
                        break
                else:
                    units_list.append('none')


        # Convert column names and units list to strings
        fields_str = ','.join(map(str, data.columns))
        units_str = ','.join(map(str, units_list))

        # Write fields (column names) and units
        f.write(f"/fields={fields_str}\n")
        f.write(f"/units={units_str}\n")

        f.write('/end_header\n')

    # Write data rows without trailing blank lines
        for _,row in data.iterrows():
            full = ''
            for col in data.columns:
                key = col
                if col.startswith('ed'):
                    key = 'ed'
                elif col.startswith('kd'):
                    key = 'kd'
                elif col.startswith('tilt'):
                    key = 'tilt'
                if pd.isna(row[col]):
                    full += missing_value_placeholder + delim
                elif key == 'ed' and row[col] < 0:
                    full += '-8888' + delim
                else:
                    full += f"{row[col]:{formatt[key]}}{delim}"
            f.write(full[:-1] +'\n')


   # data.to_csv(file_path, sep=delim, index=False, header=False, mode='a')
