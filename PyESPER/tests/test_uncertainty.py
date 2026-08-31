"""Tests for the uncertainty paths (``emlr_estimate`` / ``organize_nn_output``).

Unlike the kernel tests, these cannot use a live oracle: the implementations here are
*replaced* rather than supplemented, so there is nothing left to compare against in the
package. Reference values are therefore frozen in
``test_data/uncertainty_reference.npz``, generated from the pre-refactor code at the
branch point (``4458728``) with the inputs :func:`_inputs` builds.

These are equality checks, not tolerance checks. The refactor is arithmetic
reassociation only -- the same sums over the same float64 values in the same order --
so any difference at all is a bug, not rounding.
"""

import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REFERENCE = pathlib.Path(__file__).parent / "test_data" / "uncertainty_reference.npz"

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(),
    reason="Mat_fullgrid/ data directory not present",
)

ALL6 = ["TA", "DIC", "phosphate", "nitrate", "silicate", "oxygen"]
N = 2000


def _inputs(n=N, seed=0):
    """The exact inputs the frozen reference was generated from."""
    rng = np.random.default_rng(seed)
    coords = {
        "longitude": rng.uniform(0.0, 360.0, n),
        "latitude": rng.uniform(-78.0, 80.0, n),
        "depth": rng.uniform(0.0, 5500.0, n),
    }
    preds = {
        "salinity": rng.uniform(31.0, 37.0, n),
        "temperature": rng.uniform(-2.0, 30.0, n),
    }
    return coords, preds, np.full(n, 2010.0)


@pytest.fixture(scope="module")
def reference():
    if not _REFERENCE.is_file():
        pytest.skip(f"{_REFERENCE.name} not present")
    with np.load(_REFERENCE) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("equation", [8, 16])
@pytest.mark.parametrize("method", ["lir", "nn"])
def test_uncertainties_match_frozen_reference(method, equation, reference):
    """Uncertainty output is bit-identical to the pre-refactor implementation."""
    coords, preds, dates = _inputs()
    kwargs = dict(
        EstDates=dates, Equations=[equation], verbose=False, compute_uncertainties=True
    )
    if method == "lir":
        from PyESPER.lir import lir

        _est, _coef, unc = lir(
            ALL6, "", coords, preds, want_coefficients=False, **kwargs
        )
    else:
        from PyESPER.nn import nn

        _est, unc = nn(ALL6, "", coords, preds, **kwargs)

    assert unc is not None, f"{method} returned no uncertainties"
    for var in ALL6:
        combo = f"{var}{equation}"
        key = f"{method}__{combo}"
        assert key in reference, f"no frozen reference for {key}"
        got = np.asarray(unc[combo], dtype=np.float64).ravel()
        want = reference[key]
        assert got.shape == want.shape, f"{key}: {got.shape} != {want.shape}"
        # equal_nan: NaN is a legitimate result where a required predictor
        # uncertainty is missing, and it must land in exactly the same places.
        assert np.array_equal(got, want, equal_nan=True), (
            f"{key}: max abs diff "
            f"{np.nanmax(np.abs(got - want))!r}, "
            f"{int((~np.isclose(got, want, equal_nan=True)).sum())} of {got.size} differ"
        )


def test_uncertainties_are_none_when_disabled():
    """The gridded path relies on this: no uncertainties requested, none computed."""
    from PyESPER.nn import nn

    coords, preds, dates = _inputs(n=200)
    _est, unc = nn(
        ["TA"], "", coords, preds, EstDates=dates, Equations=[8],
        verbose=False, compute_uncertainties=False,
    )
    assert unc is None


def test_nn_uncertainty_is_positive_and_finite():
    """Sanity floor independent of the frozen values: an uncertainty is >= 0."""
    from PyESPER.nn import nn

    coords, preds, dates = _inputs(n=500)
    _est, unc = nn(
        ["TA"], "", coords, preds, EstDates=dates, Equations=[8],
        verbose=False, compute_uncertainties=True,
    )
    u = np.asarray(unc["TA8"], dtype=np.float64)
    finite = u[np.isfinite(u)]
    assert finite.size > 0
    assert (finite >= 0).all(), "negative uncertainty"
