"""Condition ESPER's input salinity toward climatology where the nets have no support.

ESPER's neural networks and LIRs were fit to GLODAPv2 bottle data, which has almost no
tropical or mid-latitude observations below ~31 PSU: the only low-salinity training
data are subpolar and Arctic shelf waters. Evaluated at a river-plume salinity (S ~ 20-28
in the Bay of Bengal, the Columbia plume, ...), the nets therefore extrapolate, and the
extrapolation is not merely imprecise but unphysical -- silicate above 100 µmol/kg,
negative nitrate/phosphate at more than half the points, and DIC above TA (measured on
12 km boundary strips of an Indo-Pacific domain, 2014: DIC > TA at 18-31 % of the surface
points with S < 31, 33-52 % in the Bay of Bengal).

This module replaces the salinity *fed to the nets* with a raised-cosine blend toward the
World Ocean Atlas 2023 annual-mean salinity climatology wherever the model salinity is
below a band::

    w  = 0                              S <= low
    w  = 0.5 - 0.5 cos(pi (S-low)/(high-low))   low < S < high
    w  = 1                              S >= high
    S' = w S + (1 - w) S_WOA(lon, lat, depth)

Above ``high`` nothing changes, bit for bit. The default band is 31-34 PSU. Only
salinity is conditioned; temperature, coordinates and dates are passed through, so the
nets are evaluated "as if the water had climatological salinity" -- which is where their
training support is. On the same 2014 strips this removed every pathology listed above
(silicate > 60 and DIC > TA to 0 %, nutrient zeros down to the rate in-range ESPER shows
in oligotrophic water) and left the in-range (S > 34) estimates untouched. The remaining
near-zero phosphate at a conditioned S ~ 31.6 in the winter Bay of Bengal agrees with the
GLODAP observations there (cell means 0.01-0.12 µmol/kg), i.e. it is what the nets were
trained on, not a defect.

**TA and DIC are diluted back to the model salinity.** Conditioning evaluates the nets at
S′, so on its own it would hand back the alkalinity and DIC of ~31.5 PSU water for a
25 PSU river plume -- but alkalinity and DIC are, to first order, conservative with
salinity (GLODAP's Bay of Bengal surface cells give TA = 43.2 S + 768, a freshwater
endmember near 770 µmol/kg, close to the global surface fit). So after estimation the
two carbonate variables are moved from S′ to S along a conservative mixing line with a
freshwater endmember::

    TA  = TA0  + (TA(S′)  - TA0)  S / S′
    DIC = DIC0 + (DIC(S′) - DIC0) S / S′

with ``TA0``/``DIC0`` configurable (both default to 770 µmol/kg, see
:data:`DEFAULT_TA_ENDMEMBER`). What this does and does not preserve:

* Above the band S == S′ and the dilution is the identity.
* Inside and below the band the DIC/TA ratio is **not** the nets' ratio: it moves from
  the nets' value at S′ toward ``DIC0/TA0`` as S → 0. With equal endmembers that limit
  is exactly 1, approached from below -- ``DIC - TA = (DIC(S′) - TA(S′)) S/S′`` keeps
  the sign the nets gave it at S′, so wherever the nets put DIC below TA (everywhere
  they have support) the diluted pair never crosses. Unequal endmembers would cross:
  with the GLODAP-fit intercepts (TA0 770, DIC0 850) DIC exceeded TA below ~8 PSU.
* Both scale with salinity; on the 2014 strips the result lands within ~100 µmol/kg of
  the GLODAP Bay of Bengal surface line extrapolated to the plume.

Nutrients and oxygen are not diluted (their freshwater endmembers are nothing like
conservative).

**The climatology file is not downloaded by this package.** Pass its path explicitly via
:class:`SalinityConditioning`; if the file is missing the error names the download URL
(:data:`WOA23_SALINITY_URL`, the 1-degree annual-mean "decav" product, ~25 MB). Callers
that manage source data (C-Star's forge source registry) are expected to fetch it and
hand the path down.

The lookup is the same construction PyESPER uses for its own LIR grids
(``PyESPER/interpolate.py``): land/missing cells are filled from the nearest valid cell on
the same depth level, then a trilinear ``RegularGridInterpolator`` over (depth, lat, lon)
is memoised process-wide in :mod:`PyESPER.kernels.grid_cache`, keyed by the file's real
path, so dask chunks share one build (~2 s).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from PyESPER.kernels import grid_cache

#: Where the 1-degree WOA23 annual-mean salinity climatology lives (NCEI). The file this
#: package expects is exactly :data:`WOA23_SALINITY_FILENAME`.
WOA23_SALINITY_URL = (
    "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/salinity/netcdf/decav/1.00/"
    "woa23_decav_s00_01.nc"
)
WOA23_SALINITY_FILENAME = "woa23_decav_s00_01.nc"

#: Variable name of the objectively analysed salinity in the WOA netCDF files.
WOA_SALINITY_VARIABLE = "s_an"

#: Default freshwater endmember (µmol/kg) for diluting TA and DIC from the conditioned
#: salinity to the model salinity. 770 is the intercept of the GLODAPv2 surface TA-S fit
#: in the Bay of Bengal (TA = 43.2 S + 768; 58 one-degree cells, S 31.4-35.2), close to
#: the global surface intercept (~590); river chemistry varies (Amazon ~300,
#: Ganges-Brahmaputra ~1700), so it is a middle value. DIC deliberately takes the SAME
#: endmember rather than its own fit intercept (855): the dilution keeps the sign of
#: DIC - TA that the nets gave at S′, so equal endmembers guarantee DIC stays below TA
#: all the way to fresh water, whereas DIC0 > TA0 made DIC exceed TA below ~8 PSU --
#: the very defect the conditioning exists to remove.
DEFAULT_TA_ENDMEMBER = 770.0
DEFAULT_DIC_ENDMEMBER = DEFAULT_TA_ENDMEMBER

#: ESPER variable names whose estimates are diluted to the model salinity.
CARBONATE_VARIABLES = ("TA", "DIC")

#: Default band in PSU: climatology outright below ``low``, model salinity outright
#: above ``high``. 31 is roughly where GLODAP's tropical/mid-latitude coverage ends; 34
#: is comfortably inside it, and the taper keeps the blended field's gradient continuous
#: at both edges.
DEFAULT_BAND = (31.0, 34.0)


@dataclass(frozen=True)
class SalinityConditioning:
    """How to condition the salinity fed to the nets. ``None`` in the APIs means off.

    Parameters
    ----------
    woa_salinity_path : path-like
        The WOA23 1-degree annual-mean salinity file (``woa23_decav_s00_01.nc``). Must
        exist; see :data:`WOA23_SALINITY_URL`.
    low, high : float
        Band edges in PSU (default 31, 34). Climatology below ``low``, model salinity
        above ``high``, raised-cosine blend between.
    """

    woa_salinity_path: str | os.PathLike
    low: float = DEFAULT_BAND[0]
    high: float = DEFAULT_BAND[1]
    #: Freshwater endmembers (µmol/kg) for the conservative dilution of TA and DIC from
    #: the conditioned salinity back to the model salinity; see the module docstring.
    #: Keep ``dic_endmember <= ta_endmember`` unless you want DIC above TA in fresh water.
    ta_endmember: float = DEFAULT_TA_ENDMEMBER
    dic_endmember: float = DEFAULT_DIC_ENDMEMBER

    def __post_init__(self) -> None:
        for name in ("ta_endmember", "dic_endmember"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative number, got {getattr(self, name)!r}.")
        if not self.high > self.low:
            raise ValueError(
                f"salinity conditioning band needs high > low, got low={self.low!r}, "
                f"high={self.high!r}."
            )
        path = Path(self.woa_salinity_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Salinity conditioning was requested but the WOA23 salinity climatology "
                f"was not found at {str(path)!r}. PyESPER does not download it; fetch "
                f"{WOA23_SALINITY_FILENAME} from {WOA23_SALINITY_URL} and pass its path "
                f"as SalinityConditioning(woa_salinity_path=...)."
            )
        # Normalise once so the cache key and any provenance string agree.
        object.__setattr__(self, "woa_salinity_path", os.path.realpath(path))

    def climatology(self, lon: ArrayLike, lat: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """WOA salinity at the points (lon °E any wrap, lat °N, depth m positive down)."""
        return woa_salinity_interpolant(self.woa_salinity_path)(lon, lat, depth)

    def apply(
        self, salinity: ArrayLike, lon: ArrayLike, lat: ArrayLike, depth: ArrayLike
    ) -> np.ndarray:
        """Conditioned salinity ``S'`` for the points (all 1-D arrays, same length)."""
        salinity = np.asarray(salinity, dtype="float64")
        w = raised_cosine_weight(salinity, self.low, self.high)
        # Skip the lookup entirely when nothing is in or below the band -- the common
        # case for open-ocean chunks, and it keeps S' == S bit for bit there.
        if np.all(w >= 1.0):
            return salinity
        out = salinity.copy()
        sel = w < 1.0
        clim = self.climatology(
            np.asarray(lon, dtype="float64")[sel],
            np.asarray(lat, dtype="float64")[sel],
            np.asarray(depth, dtype="float64")[sel],
        )
        out[sel] = w[sel] * salinity[sel] + (1.0 - w[sel]) * clim
        return out


    def dilute_carbonate(
        self,
        variable: str,
        estimate: ArrayLike,
        salinity: ArrayLike,
        conditioned: ArrayLike,
    ) -> np.ndarray:
        """Move a TA/DIC estimate made at ``conditioned`` salinity to the model
        ``salinity`` along a conservative mixing line (see the module docstring).

        Other variables are returned unchanged, as is every point where the two
        salinities agree (above the band), so the identity holds bit for bit there.
        """
        if variable not in CARBONATE_VARIABLES:
            return estimate
        endmember = self.ta_endmember if variable == "TA" else self.dic_endmember
        return dilute_to_salinity(estimate, salinity, conditioned, endmember)


def dilute_to_salinity(
    estimate: ArrayLike, salinity: ArrayLike, conditioned: ArrayLike, endmember: float
) -> np.ndarray:
    """``endmember + (estimate - endmember) * salinity / conditioned`` where the two
    salinities differ; ``estimate`` itself elsewhere (and wherever ``conditioned`` is 0)."""
    estimate = np.asarray(estimate, dtype="float64")
    salinity = np.asarray(salinity, dtype="float64")
    conditioned = np.asarray(conditioned, dtype="float64")
    out = estimate.copy()
    sel = (salinity != conditioned) & (conditioned > 0)
    if sel.any():
        out[sel] = endmember + (estimate[sel] - endmember) * salinity[sel] / conditioned[sel]
    return out


def raised_cosine_weight(salinity: ArrayLike, low: float, high: float) -> np.ndarray:
    """Weight of the *model* salinity: 0 at/below ``low``, 1 at/above ``high``.

    ``0.5 - 0.5 cos(pi x)`` on the normalised band coordinate ``x`` (a Tukey taper): its
    derivative vanishes at both edges, so the conditioned field keeps a continuous
    gradient where the band starts and ends; a linear ramp would not. NaN in -> NaN out.
    """
    x = np.clip((np.asarray(salinity, dtype="float64") - low) / (high - low), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


class _WOAInterpolant:
    """Trilinear lookup into the land-filled WOA salinity cube; lon wraps, depth clamps."""

    def __init__(
        self, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray, values: np.ndarray
    ) -> None:
        from scipy.interpolate import RegularGridInterpolator

        # Pad longitude periodically so queries between the last and first column
        # interpolate across the seam instead of extrapolating.
        lon_p = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
        values_p = np.concatenate((values[:, :, -1:], values, values[:, :, :1]), axis=2)
        self._depth_max = float(depth[-1])
        self._lat_min, self._lat_max = float(lat[0]), float(lat[-1])
        self._lon_lo = float(lon_p[0])
        self._f = RegularGridInterpolator(
            (depth, lat, lon_p), values_p, method="linear",
            bounds_error=False, fill_value=None,
        )

    def __call__(self, lon: ArrayLike, lat: ArrayLike, depth: ArrayLike) -> np.ndarray:
        lon = np.asarray(lon, dtype="float64")
        lon = self._lon_lo + np.mod(lon - self._lon_lo, 360.0)
        lat = np.clip(np.asarray(lat, dtype="float64"), self._lat_min, self._lat_max)
        # Below the deepest level use the deepest level; above 0 m (never, but cheap) 0.
        depth = np.clip(np.asarray(depth, dtype="float64"), 0.0, self._depth_max)
        return self._f(np.column_stack((depth, lat, lon)))


def _build_interpolant(path: str) -> _WOAInterpolant:
    import xarray as xr
    from scipy.interpolate import NearestNDInterpolator

    with xr.open_dataset(path, decode_times=False) as ds:
        if WOA_SALINITY_VARIABLE not in ds:
            raise ValueError(
                f"{path!r} has no {WOA_SALINITY_VARIABLE!r} variable; expected the WOA "
                f"salinity climatology {WOA23_SALINITY_FILENAME} ({WOA23_SALINITY_URL})."
            )
        da = ds[WOA_SALINITY_VARIABLE]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.transpose("depth", "lat", "lon")
        values = np.asarray(da.values, dtype="float64")
        depth = np.asarray(da["depth"].values, dtype="float64")
        lat = np.asarray(da["lat"].values, dtype="float64")
        lon = np.asarray(da["lon"].values, dtype="float64")

    # Fill land / below-seafloor cells from the nearest valid cell on the same level,
    # with the longitude seam handled by querying against a periodically extended copy.
    lon_ext = np.concatenate((lon - 360.0, lon, lon + 360.0))
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    la3, lo3 = np.meshgrid(lat, lon_ext, indexing="ij")
    for k in range(values.shape[0]):
        level = values[k]
        good = np.isfinite(level)
        if good.all():
            continue
        if not good.any():
            # A level with no ocean at all (does not occur in WOA): carry the level above.
            values[k] = values[k - 1] if k else 0.0
            continue
        good3 = np.tile(good, (1, 3))
        level3 = np.tile(level, (1, 3))
        fill = NearestNDInterpolator(
            np.column_stack((la3[good3], lo3[good3])), level3[good3]
        )
        level[~good] = fill(np.column_stack((la[~good], lo[~good])))
    return _WOAInterpolant(depth, lat, lon, values)


def woa_salinity_interpolant(path: str | os.PathLike) -> _WOAInterpolant:
    """The memoised lookup for ``path`` (built on first use, shared process-wide)."""
    return grid_cache.woa_salinity(path, lambda p: _build_interpolant(p))
