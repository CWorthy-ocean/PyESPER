import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

_interpolator_cache = {}


def simplecantestimatelr(EstDates, longitude, latitude, depth, data_path=None):
    global _interpolator_cache

    if data_path is None:
        data_path = Path(__file__).parent / "SimpleCantEstimateLR_full.csv"
    else:
        data_path = Path(data_path)

    cache_key = str(data_path)

    if cache_key not in _interpolator_cache:

        CantIntPoints = pd.read_csv(data_path)

        u_lon, lon_idx = np.unique(
            CantIntPoints["Int_long"], return_inverse=True
        )
        u_lat, lat_idx = np.unique(
            CantIntPoints["Int_lat"], return_inverse=True
        )
        u_depth, depth_idx = np.unique(
            CantIntPoints["Int_depth"], return_inverse=True
        )

        grid_values = np.empty((len(u_depth), len(u_lat), len(u_lon)))
        grid_values[depth_idx, lat_idx, lon_idx] = CantIntPoints["values"]

        _interpolator_cache[cache_key] = RegularGridInterpolator(
            (u_depth * 0.025, u_lat, u_lon * 0.25),
            grid_values,
            bounds_error=False,
            fill_value=np.nan,
        )

    pointso = np.column_stack(
        (
            np.asarray(depth) * 0.025,
            np.asarray(latitude),
            np.asarray(longitude) * 0.25,
        )
    )

    Cant2002 = _interpolator_cache[cache_key](pointso)

    EstDates = np.asarray(EstDates)
    CantMeas = Cant2002 * np.exp(0.018989 * (EstDates - 2002.0))

    return CantMeas, Cant2002
