"""Dask-lazy, xarray-native wrappers around the PyESPER estimation routines.

The stock :func:`PyESPER.lir.lir` / :func:`PyESPER.nn.nn` / :func:`PyESPER.mixed.mixed`
entry points take dict-of-1-D-list inputs and return dict-of-1-D-array estimates for a
flat set of points. They are fully vectorised over the points and (with
``compute_uncertainties=False``) have modest, roughly fixed overhead per call.

Seawater property estimation is **point-independent**: each location's estimate depends
only on its own longitude/latitude/depth/salinity/temperature(/date). That makes it a
natural fit for :func:`xarray.apply_ufunc` with ``dask="parallelized"`` -- dask invokes
the kernel on one chunk at a time, so peak memory is bounded by the chunk size (not the
whole domain or the number of time records), and independent chunks run in parallel.

Public functions
----------------
``lir_xr`` / ``nn_xr`` / ``mixed_xr`` accept :class:`xarray.DataArray` inputs and return a
dict ``{variable: DataArray}`` of estimates (µmol/kg, ESPER's native units) with the same
dims/coords as the broadcast inputs. They compute **estimates only** (no uncertainties),
so downstream code can assign the results lazily to a dataset and only materialise them at
write time.

Notes
-----
* NaN inputs (e.g. land cells on a model grid) are handled: non-finite points are skipped
  and returned as NaN, so the underlying routines only ever see valid points.
* The neural-net path (``nn_xr``/``mixed_xr``) is safe under a process/synchronous dask
  scheduler. The reload-based module loading was removed (see ``run_nets._load_net``), so
  the remaining thread-safety concern is Numba first-compilation; prefer the process or
  synchronous scheduler, or warm the caches once before fanning out.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# ESPER's estimable variables (PyESPER naming).
VALID_VARIABLES = ("TA", "DIC", "pH", "phosphate", "nitrate", "silicate", "oxygen")

# The predictor set these wrappers support: coordinates + salinity + temperature only,
# i.e. Equation 8 (S, T) or 16 (S). Extending to nutrient/oxygen predictors would mean
# threading those DataArrays through as additional PredictorMeasurements.
_METHODS = {"lir", "nn", "mixed"}


def _method_fn(method):
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted(_METHODS)}, got {method!r}."
        )
    if method == "lir":
        from PyESPER.lir import lir

        def _call(variables, path, coords, preds, dates, equation):
            est, _coef, _unc = lir(
                variables, path, coords, preds,
                EstDates=dates, Equations=[equation],
                verbose=False, compute_uncertainties=False,
            )
            return est

        return _call
    if method == "nn":
        from PyESPER.nn import nn

        def _call(variables, path, coords, preds, dates, equation):
            est, _unc = nn(
                variables, path, coords, preds,
                EstDates=dates, Equations=[equation],
                verbose=False, compute_uncertainties=False,
            )
            return est

        return _call

    from PyESPER.mixed import mixed

    def _call(variables, path, coords, preds, dates, equation):
        est, _unc = mixed(
            variables, path, coords, preds,
            EstDates=dates, Equations=[equation],
            verbose=False, compute_uncertainties=False,
        )
        return est

    return _call


def _estimate_block(sal, temp, lon, lat, depth, dates, *, variables, path, method, equation):
    """Estimate ``variables`` for one (numpy) block of points.

    All inputs are numpy arrays of an identical arbitrary shape (the dask block). Returns
    a tuple of arrays (one per variable, in ``variables`` order), each the same shape as
    the inputs, in µmol/kg. Non-finite points are returned as NaN.
    """
    shape = sal.shape
    n_out = len(variables)
    outs = [np.full(shape, np.nan, dtype="float64") for _ in range(n_out)]

    sal_f = np.asarray(sal, dtype="float64").ravel()
    temp_f = np.asarray(temp, dtype="float64").ravel()
    lon_f = np.asarray(lon, dtype="float64").ravel()
    lat_f = np.asarray(lat, dtype="float64").ravel()
    depth_f = np.asarray(depth, dtype="float64").ravel()
    dates_f = np.asarray(dates, dtype="float64").ravel()

    valid = (
        np.isfinite(sal_f) & np.isfinite(temp_f) & np.isfinite(lon_f)
        & np.isfinite(lat_f) & np.isfinite(depth_f) & np.isfinite(dates_f)
    )
    if not valid.any():
        return tuple(outs)

    idx = np.flatnonzero(valid)
    coords = {
        "longitude": lon_f[idx].tolist(),
        "latitude": lat_f[idx].tolist(),
        "depth": depth_f[idx].tolist(),
    }
    preds = {
        "salinity": sal_f[idx].tolist(),
        "temperature": temp_f[idx].tolist(),
    }
    est = _method_fn(method)(
        list(variables), path, coords, preds, dates_f[idx].tolist(), equation
    )

    flat_outs = [np.full(sal_f.shape, np.nan, dtype="float64") for _ in range(n_out)]
    for i, var in enumerate(variables):
        key = f"{var}{equation}"
        flat_outs[i][idx] = np.asarray(est[key], dtype="float64").ravel()
        outs[i] = flat_outs[i].reshape(shape)
    return tuple(outs)


def _estimate_xr(salinity, temperature, longitude, latitude, depth, *,
                 variables, path, method, equation, est_dates):
    """Shared implementation for ``lir_xr``/``nn_xr``/``mixed_xr``."""
    if isinstance(variables, str):
        variables = [variables]
    variables = list(variables)
    unknown = [v for v in variables if v not in VALID_VARIABLES]
    if unknown:
        raise ValueError(
            f"Unknown variable(s) {unknown}; valid options: {list(VALID_VARIABLES)}."
        )
    if equation not in (8, 16):
        raise ValueError(
            "equation must be 8 (salinity+temperature) or 16 (salinity only); "
            f"got {equation!r}. These are the S/T-only ESPER equations."
        )

    # Dates as a DataArray so it broadcasts alongside the fields (decimal year).
    if est_dates is None:
        est_dates = 2002.0
    if not isinstance(est_dates, xr.DataArray):
        est_dates = xr.DataArray(est_dates)

    # Align/broadcast every input to a common set of dims (lazy for dask inputs).
    sal, temp, lon, lat, dep, dates = xr.broadcast(
        salinity, temperature, longitude, latitude, depth, est_dates
    )

    results = xr.apply_ufunc(
        _estimate_block,
        sal, temp, lon, lat, dep, dates,
        kwargs=dict(
            variables=variables, path=str(path), method=method, equation=equation
        ),
        output_core_dims=[[]] * len(variables),
        dask="parallelized",
        output_dtypes=[np.float64] * len(variables),
    )
    if len(variables) == 1:
        results = (results,)
    return {var: results[i] for i, var in enumerate(variables)}


def lir_xr(salinity, temperature, longitude, latitude, depth, *,
           variables, path="", equation=8, est_dates=None):
    """Dask-lazy LIR estimates as xarray DataArrays. See module docstring.

    Parameters
    ----------
    salinity, temperature, longitude, latitude, depth : xarray.DataArray
        Predictors/coordinates on the target grid (any shared/broadcastable dims). Salinity
        is PSS-78, temperature in-situ °C, depth metres (positive down), longitude °E,
        latitude °N.
    variables : str or list of str
        ESPER variable name(s) to estimate: subset of ``TA``, ``DIC``, ``phosphate``,
        ``nitrate``, ``silicate``, ``oxygen``, ``pH``.
    path : str
        Directory containing the ESPER data (``Mat_fullgrid/`` etc.).
    equation : int
        8 (salinity + temperature) or 16 (salinity only).
    est_dates : float or xarray.DataArray, optional
        Decimal year(s); only affects DIC/pH. Defaults to 2002.0.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{variable: estimate}`` in µmol/kg, dims/coords matching the broadcast inputs.
    """
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="lir", equation=equation,
        est_dates=est_dates,
    )


def nn_xr(salinity, temperature, longitude, latitude, depth, *,
          variables, path="", equation=8, est_dates=None):
    """Dask-lazy neural-network estimates as xarray DataArrays. See :func:`lir_xr`."""
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="nn", equation=equation,
        est_dates=est_dates,
    )


def mixed_xr(salinity, temperature, longitude, latitude, depth, *,
             variables, path="", equation=8, est_dates=None):
    """Dask-lazy LIR+NN ensemble-mean estimates as xarray DataArrays. See :func:`lir_xr`."""
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="mixed", equation=equation,
        est_dates=est_dates,
    )
