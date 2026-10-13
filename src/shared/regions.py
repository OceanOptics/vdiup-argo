from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np

DEFAULT_SEAS_PATH = Path('data') / 'World_seas_IHO_v3' / 'World_Seas_IHO_v3.shp'


@lru_cache(maxsize=2)
def _load_seas(path):
    return gpd.read_file(path)[["NAME", "geometry"]]


def get_region_name(lat, lon, seas_path=DEFAULT_SEAS_PATH, name_column="NAME"):
    """
    Return the IHO sea/ocean area name containing each GPS coordinate.

    Uses a vectorized spatial join (R-tree index), so arrays of any size
    are handled efficiently. The shapefile is read once and cached.
    Shapefile: World_Seas_IHO_v3 from Marine Regions (https://www.marineregions.org).

    Example:
        >>> get_region_name(70.0, -65.0)
        'Baffin Bay'
        >>> get_region_name([70.0, 36.0, 0.0], [-65.0, 15.0, -30.0])
        array(['Baffin Bay', 'Mediterranean Region', 'North Atlantic Ocean'], dtype=object)

    Parameters
    ----------
    lat, lon : float or array-like
        Latitude / longitude in decimal degrees (same shape, broadcastable).
    seas_path : str
        Path to the IHO World Seas shapefile.
    name_column : str
        Attribute column holding the region name.

    Returns
    -------
    str or numpy.ndarray of object
        Region name (None where no region contains the point, e.g. on land
        or NaN input). A str is returned for scalar input, otherwise an
        array with the broadcast shape of the inputs.
    """
    scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    shape = lat.shape
    lat, lon = lat.ravel(), lon.ravel()

    names = np.full(lat.shape, None, dtype=object)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.any():
        seas = _load_seas(seas_path)
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lon[valid], lat[valid]), crs=seas.crs
        )
        joined = points.sjoin(seas, how="left", predicate="covered_by")
        # Boundary points can match several polygons: keep the first match
        joined = joined[~joined.index.duplicated(keep="first")].sort_index()
        found = joined[name_column].astype(object).where(joined[name_column].notna(), None)
        names[np.flatnonzero(valid)] = found.to_numpy()

    names = names.reshape(shape)
    return names.item() if scalar else names
