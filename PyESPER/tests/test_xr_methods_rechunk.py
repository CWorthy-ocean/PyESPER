"""Regression test for the defensive rechunking in ``xr_methods._estimate_xr``.

Context: a caller's own dask chunking (e.g. a physics grid regridded with the
entire vertical dimension in one chunk) can produce chunks of tens of millions
of points. ``run_nets``' own memory cost is ~10 KB/point (dominated by
``batched_forward``'s (G nets, hidden width, Q points) activation arrays), so an
unbounded chunk size there is not "fine, dask handles it lazily" -- it is a
real OOM risk (this exact scenario OOM-killed a 251 GB machine twice on a 4 km
/ 100-level production grid before the fix this test pins). This test does not
run the actual neural nets (expensive); it only verifies that no chunk handed
to ``_estimate_block`` ever exceeds ``_MAX_POINTS_PER_CHUNK``, regardless of
how coarsely the input arrays were chunked by the caller.
"""

from unittest.mock import patch

import numpy as np
import xarray as xr

from PyESPER.xr_methods import _MAX_POINTS_PER_CHUNK, nn_xr


def _fake_estimate_block(sal, temp, lon, lat, depth, dates, *, variables, path, method, equation):
    """Records the chunk's point count instead of calling any real net."""
    _fake_estimate_block.chunk_sizes.append(sal.size)
    return tuple(np.zeros(sal.shape, dtype="float64") for _ in variables)


_fake_estimate_block.chunk_sizes = []


def test_large_input_chunk_gets_rechunked_below_budget():
    """A caller-supplied chunk far bigger than the budget must be split before
    reaching the per-chunk kernel -- not passed through as-is.
    """
    _fake_estimate_block.chunk_sizes = []

    # Mimic the real production shape that caused the crash: one big chunk
    # along a "vertical" dim, a few chunks horizontally -- total chunk size
    # (100 x 400 x 400 = 16,000,000) is ~80x over budget.
    shape = (100, 800, 800)
    rng = np.random.default_rng(0)
    sal = xr.DataArray(
        rng.uniform(32, 36, shape), dims=("s", "y", "x")
    ).chunk({"s": 100, "y": 400, "x": 400})
    temp = xr.DataArray(rng.uniform(2, 20, shape), dims=("s", "y", "x")).chunk(sal.chunksizes)
    lon = xr.DataArray(rng.uniform(-180, -110, shape), dims=("s", "y", "x")).chunk(sal.chunksizes)
    lat = xr.DataArray(rng.uniform(20, 55, shape), dims=("s", "y", "x")).chunk(sal.chunksizes)
    depth = xr.DataArray(rng.uniform(0, 500, shape), dims=("s", "y", "x")).chunk(sal.chunksizes)

    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            sal, temp, lon, lat, depth,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8, est_dates=2013.0,
        )
        # Force the lazy apply_ufunc graph to actually run the (mocked) kernel.
        out["nitrate"].compute(scheduler="synchronous")

    assert _fake_estimate_block.chunk_sizes, "kernel was never invoked"
    assert max(_fake_estimate_block.chunk_sizes) <= _MAX_POINTS_PER_CHUNK, (
        f"a chunk of {max(_fake_estimate_block.chunk_sizes):,} points reached the "
        f"kernel -- over the {_MAX_POINTS_PER_CHUNK:,}-point safety budget"
    )
    # Sanity: the rechunk must actually have split the array into more than
    # the original handful of chunks, not silently left it alone.
    assert len(_fake_estimate_block.chunk_sizes) > 6


def test_already_small_chunks_are_left_alone_in_count():
    """Inputs already within budget should not be needlessly fragmented further."""
    _fake_estimate_block.chunk_sizes = []

    shape = (10, 50, 50)  # 25,000 points total, well under budget
    rng = np.random.default_rng(0)
    sal = xr.DataArray(rng.uniform(32, 36, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    temp = xr.DataArray(rng.uniform(2, 20, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    lon = xr.DataArray(rng.uniform(-180, -110, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    lat = xr.DataArray(rng.uniform(20, 55, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    depth = xr.DataArray(rng.uniform(0, 500, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})

    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            sal, temp, lon, lat, depth,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8, est_dates=2013.0,
        )
        out["nitrate"].compute(scheduler="synchronous")

    assert _fake_estimate_block.chunk_sizes == [shape[0] * shape[1] * shape[2]]


def test_numpy_backed_inputs_are_unaffected():
    """Non-dask (eager numpy) inputs must not be forced into dask by this path."""
    n = 100
    rng = np.random.default_rng(0)
    sal = xr.DataArray(rng.uniform(32, 36, n), dims=("p",))
    temp = xr.DataArray(rng.uniform(2, 20, n), dims=("p",))
    lon = xr.DataArray(rng.uniform(-180, -110, n), dims=("p",))
    lat = xr.DataArray(rng.uniform(20, 55, n), dims=("p",))
    depth = xr.DataArray(rng.uniform(0, 500, n), dims=("p",))

    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            sal, temp, lon, lat, depth,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8, est_dates=2013.0,
        )

    assert not hasattr(out["nitrate"].data, "dask")
