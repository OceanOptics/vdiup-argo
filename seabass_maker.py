import pandas as pd
import numpy as np


def format_to_seabass(data, metadata, filename, missing_value_placeholder='-9999', delimiter='comma'):
    """
    Formats the data and metadata into NASA SeaBASS format and writes to a .sb file.

    Parameters:
    - data (pd.DataFrame): The dataset to be formatted.
    - metadata (dict): A dictionary containing metadata information.
    - filename (str): The output file name without extension.
    - missing_value_placeholder (str): The placeholder for missing values.
    - delimiter (str): The delimiter to use ('comma', 'tab', or 'space').
    """
    # Ensure the filename has the correct extension
    if not filename.endswith('.sb'):
        filename += '.sb'

    # Handle missing values in datetime columns separately
    datetime_cols = data.select_dtypes(include=[np.datetime64]).columns
    data[datetime_cols] = data[datetime_cols].fillna(pd.Timestamp('19000101'))

    # Replace missing values in other columns with '-9999'
    other_cols = data.columns.difference(datetime_cols)
    data[other_cols] = data[other_cols].fillna(missing_value_placeholder)

    # Determine the delimiter
    if delimiter == 'comma':
        delim = ','
    elif delimiter == 'tab':
        delim = '\t'
    elif delimiter == 'space':
        delim = ' '
    else:
        raise ValueError("Unsupported delimiter. Choose from 'comma', 'tab', or 'space'.")

    with open(filename, 'w') as f:
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
        start_date = data['TIME'].min()[:10].replace('-', '')
        end_date = data['TIME'].max()[:10].replace('-', '')
        start_time = data['TIME'].min()[11:19]
        end_time = data['TIME'].max()[11:19]
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
        f.write('/end_header\n')

        # Write column names
        f.write(delim.join(data.columns) + '\n')

        # Write data rows
        data.to_csv(f, sep=delim, index=False, header=False)

# Define metadata for the SeaBASS file
metadata = {
    'investigators': 'Nils_Haentjens, Charlotte_Begouen_Demeaux',
    'affiliations': 'University_of_Maine, University_of_Maine',
    'contact': 'nils.haentjens@maine.edu',
    'experiment': 'PVST-VDIUP',
    'cruise': 'BGC_' +'wmo',
    'platform_id': 'wmo',
    'instrument_manufacturer': 'TriOS',
    'instrument_model': 'Ramses',
    'profile': 'cycle_number',
    'documents': 'none',
    'calibration_files': 'none',
    'data_type': 'Drifter',
    'data_status': 'preliminary',
    'water_depth': 'NA',
    'measurement_depth': 'NA',
}
