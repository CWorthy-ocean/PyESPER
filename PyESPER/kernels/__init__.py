"""Fused, tiled numba kernels for PyESPER.

Each module here holds one kernel plus the packing/caching it needs. The kernels are
written so that a single ``prange`` over point *tiles* is the only parallelism in play
(see :mod:`PyESPER.concurrency`), and so that every intermediate lives in per-tile
scratch rather than in an n_points-sized array.
"""
