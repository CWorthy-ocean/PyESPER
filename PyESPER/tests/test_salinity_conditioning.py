"""Tests for :mod:`PyESPER.salinity_conditioning` and its hook in ``xr_methods``.

They use a small synthetic climatology written to ``tmp_path`` (same layout as the WOA
file: ``s_an(time, depth, lat, lon)`` with NaN over land, 1-degree cell centres), so no
real data or nets are needed. The nets themselves are replaced by a fake that echoes the
salinity it was handed, which is exactly the thing these tests need to observe.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from PyESPER.kernels import grid_cache
from PyESPER.salinity_conditioning import (
    WOA23_SALINITY_URL,
    SalinityConditioning,
    raised_cosine_weight,
    woa_salinity_interpolant,
)
from PyESPER.xr_methods import nn_xr


@pytest.fixture(autouse=True)
def _clear_cache():
    grid_cache.clear()
    yield
    grid_cache.clear()


@pytest.fixture
def woa_file(tmp_path):
    """s_an = 30 + depth/100 + lat/90 on ocean cells; NaN on a land block and the
    seam column lon=179.5, so the fill and the wrap are both exercised."""
    depth = np.array([0.0, 50.0, 200.0])
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    d, la, lo = np.meshgrid(depth, lat, lon, indexing="ij")
    s = 30.0 + d / 100.0 + la / 90.0
    land = (lat[None, :, None] > 0) & (lat[None, :, None] < 10) & (lon[None, None, :] > 0) & (lon[None, None, :] < 10)
    s = np.where(np.broadcast_to(land, s.shape), np.nan, s)
    s[:, :, -1] = np.nan  # the 179.5 column
    ds = xr.Dataset(
        {"s_an": (("time", "depth", "lat", "lon"), s[None].astype("float32"))},
        coords={"time": [0.0], "depth": depth, "lat": lat, "lon": lon},
    )
    path = tmp_path / "woa23_decav_s00_01.nc"
    ds.to_netcdf(path)
    return path


def test_missing_file_names_the_download_url(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        SalinityConditioning(tmp_path / "nope.nc")
    assert WOA23_SALINITY_URL in str(info.value)
    assert "does not download" in str(info.value)


def test_band_must_be_ordered(woa_file):
    with pytest.raises(ValueError):
        SalinityConditioning(woa_file, low=34, high=31)


def test_weight_is_a_raised_cosine():
    w = raised_cosine_weight(np.array([20.0, 31.0, 32.5, 34.0, 36.0, np.nan]), 31.0, 34.0)
    np.testing.assert_allclose(w[:5], [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.isnan(w[5])


def test_lookup_fills_land_and_wraps_longitude(woa_file):
    f = woa_salinity_interpolant(woa_file)
    # Ocean cell centre: exact formula value.
    np.testing.assert_allclose(f([-30.5], [40.5], [50.0]), [30.0 + 0.5 + 40.5 / 90.0])
    # Land block interior: filled from a neighbour on the same level -> finite, in range.
    v = f([5.5], [5.5], [0.0])
    assert np.isfinite(v).all() and 29.0 < v[0] < 31.0
    # The seam column was NaN in the file; after the fill, lon 179.5 == lon -180.5 == 539.5.
    a, b, c = f([179.5], [-20.5], [0.0]), f([-180.5], [-20.5], [0.0]), f([539.5], [-20.5], [0.0])
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, c)
    # Below the deepest level: clamp to it rather than extrapolate.
    np.testing.assert_allclose(f([-30.5], [40.5], [5000.0]), f([-30.5], [40.5], [200.0]))


def test_apply_blends_only_below_the_band(woa_file):
    cond = SalinityConditioning(woa_file)
    lon, lat, dep = np.full(4, -30.5), np.full(4, 40.5), np.full(4, 50.0)
    sal = np.array([20.0, 32.5, 34.0, 35.5])
    clim = 30.0 + 0.5 + 40.5 / 90.0
    out = cond.apply(sal, lon, lat, dep)
    np.testing.assert_allclose(out, [clim, 0.5 * 32.5 + 0.5 * clim, 34.0, 35.5])
    # Entirely above the band: returned untouched (and no lookup is built).
    grid_cache.clear()
    out = cond.apply(np.array([34.0, 36.0]), lon[:2], lat[:2], dep[:2])
    np.testing.assert_array_equal(out, [34.0, 36.0])
    assert grid_cache.cache_info()["woa_salinity_entries"] == 0


def _echo_salinity(variables, path, coords, preds, dates, equation):
    """Fake net: every variable returns the salinity it was handed."""
    return {f"{v}{equation}": np.asarray(preds["salinity"]) for v in variables}


def _inputs(chunk=None):
    sal = xr.DataArray(np.array([[20.0, 32.5], [35.5, np.nan]]), dims=("y", "x"))
    temp = xr.full_like(sal, 15.0)
    lon = xr.full_like(sal, -30.5)
    lat = xr.full_like(sal, 40.5)
    dep = xr.full_like(sal, 50.0)
    arrays = (sal, temp, lon, lat, dep)
    if chunk:
        arrays = tuple(a.chunk(chunk) for a in arrays)
    return arrays


@pytest.mark.parametrize("chunk", [None, {"y": 1, "x": 1}])
def test_nn_xr_passes_conditioned_salinity_to_the_nets(woa_file, chunk):
    cond = SalinityConditioning(woa_file)
    clim = 30.0 + 0.5 + 40.5 / 90.0
    with patch("PyESPER.xr_methods._method_fn", return_value=_echo_salinity):
        off = nn_xr(*_inputs(chunk), variables="nitrate", path="/nonexistent", est_dates=2014.0)
        on = nn_xr(*_inputs(chunk), variables="nitrate", path="/nonexistent", est_dates=2014.0,
                   salinity_conditioning=cond)
        off, on = off["nitrate"].compute(scheduler="synchronous"), on["nitrate"].compute(scheduler="synchronous")
    np.testing.assert_array_equal(off.values, [[20.0, 32.5], [35.5, np.nan]])
    np.testing.assert_allclose(on.values, [[clim, 0.5 * 32.5 + 0.5 * clim], [35.5, np.nan]])
    # One lookup per process, however many chunks.
    assert grid_cache.cache_info()["woa_salinity_entries"] == 1


def test_nn_xr_rejects_a_bare_path(woa_file):
    with pytest.raises(TypeError, match="SalinityConditioning"):
        nn_xr(*_inputs(), variables="nitrate", path="/nonexistent",
              salinity_conditioning=str(woa_file))


def test_cache_keyed_by_real_path(woa_file, tmp_path):
    link = tmp_path / "alias.nc"
    link.symlink_to(woa_file)
    woa_salinity_interpolant(woa_file)
    woa_salinity_interpolant(link)
    assert grid_cache.cache_info()["woa_salinity_entries"] == 1
