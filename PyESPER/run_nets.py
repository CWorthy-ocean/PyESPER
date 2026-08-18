import functools

import numba
import numpy as np

from PyESPER.net_weights import architecture_signature, parse_net_weights


@functools.lru_cache(maxsize=None)
def _load_net(module_name):
    """Import a hard-coded ESPER neural-net weight module once and return its
    ``PyESPER_NN`` forward-pass function.

    The ``NeuralNetworks.ESPER_*`` modules are static (hard-coded weight matrices),
    so importing them once per process and caching is both correct and much faster
    than the original per-call ``importlib.reload``. Removing ``reload`` also makes
    this path thread-safe (``importlib.reload`` mutates ``sys.modules`` and is not),
    which matters when the estimate is driven per dask chunk.

    Not used by ``run_nets`` itself any more (see module docstring) -- kept as a
    small, self-contained utility for anything that wants a single net's callable
    directly (e.g. ad hoc debugging), and so this stays a minimal diff.
    """
    import importlib

    return importlib.import_module(module_name).PyESPER_NN


@numba.njit(parallel=True, fastmath=True, cache=True)
def _tansig_kernel(flat_x, flat_out):
    for i in numba.prange(flat_x.shape[0]):
        flat_out[i] = 2.0 / (1.0 + np.exp(-2.0 * flat_x[i])) - 1.0
    return flat_out


def _tansig(x):
    """The MATLAB ``tansig`` transfer function, applied to every hidden layer's
    output (every layer except the last -- see ``batched_forward``).

    Numba-JIT'd and parallelized across a flat view of ``x``: measured 7-8x
    faster than the equivalent plain-numpy expression
    (``2 / (1 + np.exp(-2 * x)) - 1``) at the array sizes this actually runs at
    (hundreds of millions of elements -- e.g. 12 stacked nets x 40 hidden units
    x 1M points), bit-identical output. Plain numpy pays for a fresh temporary
    array at every one of the 5 operations in that expression; fusing them into
    one JIT'd loop (no intermediate arrays) accounts for roughly half the
    speedup, and letting numba parallelize the loop across cores (``prange``,
    not just ``njit``) accounts for the rest -- see the benchmark this is
    based on for the breakdown. ``np.tanh`` -- algebraically identical to
    ``tansig`` (verified: max abs difference ~3e-16, floating-point-epsilon
    level) and the "obvious" built-in alternative -- was measured *slower*
    than even the original plain-numpy expression on this system; don't
    "simplify" to it without re-benchmarking on whatever system that change is
    proposed for.

    Thread-safety / oversubscription note: this function's own internal
    parallelism (via ``prange``) claims up to ``numba.get_num_threads()``
    threads *on top of* whatever else is already running concurrently. Calling
    it from multiple threads at once (e.g. under dask's default threaded
    scheduler, several chunks in flight together) multiplies that -- the same
    N-workers-x-N-threads oversubscription ``cstar_forge`` already guards
    against for BLAS (see its ``executor.py``/``input_data.py``, which pins BLAS
    to 1 thread and caps dask's worker count for exactly this reason) and that
    ``PyESPER/xr_methods.py``'s module docstring already flags as a numba
    first-compilation thread-safety concern for this exact neural-net path.
    Run ESPER generation under a synchronous/serial dask scheduler (one chunk
    at a time) so this function's own parallelism is the only parallelism in
    play, rather than trying to have both at once.
    """
    # `.ravel()` only avoids a copy when `x` is already contiguous; `x` always
    # is in practice here (fresh output of `matmul` + broadcast-add, never a
    # transpose/slice view), but `np.ascontiguousarray` makes that a guarantee
    # rather than an assumption -- a no-op when it already holds, so this
    # costs nothing in the actual call path.
    flat_x = np.ascontiguousarray(x).ravel()
    flat_out = np.empty_like(flat_x)
    _tansig_kernel(flat_x, flat_out)
    return flat_out.reshape(x.shape)


def batched_forward(weights_list, x_stack):
    """Evaluate G structurally-identical nets against G (possibly distinct)
    predictor arrays in one vectorized pass, instead of G separate Python-level
    calls each doing their own ``mapminmax``/``tansig``/matmul over the full point
    count.

    Parameters
    ----------
    weights_list : list of dict
        G weight dicts from :func:`PyESPER.net_weights.parse_net_weights`, all
        sharing the same :func:`PyESPER.net_weights.architecture_signature` (same
        number of layers, same per-layer shapes) -- required for the per-layer
        ``np.stack`` calls below to succeed; nets with a different architecture
        must go through a separate call (see ``run_nets``, which groups by
        signature before calling this).
    x_stack : ndarray, shape (G, n_in, Q)
        Each net's own predictor array. Not necessarily identical across the G
        nets: two nets for the *same* physical points can legitimately want
        different inputs here -- e.g. the Atlantic/Other regional pair for one
        ESPER target variable share their input (same points, same predictor
        columns), but two *different* target variables (say "oxygen" vs
        "nitrate") assign the same underlying measurements to the S/T/A/B/C
        predictor slots differently (see ``iterations.py``'s ``NeededForProperty``/
        ``VarVec``), so their per-net ``X`` genuinely differs even though every
        net here shares one architecture. Passing a full (G, n_in, Q) array (with
        genuine duplication where a net's input happens to be identical to
        another's, e.g. the Atl/Other case) keeps this function simple and
        correct regardless of which case applies, at the cost of holding G copies
        of a (n_in, Q) array rather than fewer -- negligible next to the (G, Q)
        output and the point-count-sized arrays already required.

    Returns
    -------
    ndarray, shape (G, Q)
        One row per input net, in ``weights_list`` order.

    Notes
    -----
    This computes the *exact same arithmetic*, in the *exact same order*
    (subtract, multiply, add for ``mapminmax``; weighted-sum-then-bias-then-
    ``tansig`` per layer), as calling each of the G modules'
    ``PyESPER_NN`` one at a time -- only reorganized so the G nets' matching
    operations become one array op apiece instead of G. Differences from the
    original per-net path should be at the floating-point-non-associativity
    level (a different summation order inside the underlying BLAS matmul) --
    see ``PyESPER/tests/test_run_nets_batched.py``, which checks this directly
    against the original (still-present, still-importable) per-module
    ``PyESPER_NN`` functions.
    """
    n_layers = len(weights_list[0]["layers"])

    x1_xoffset = np.stack([w["x1_xoffset"] for w in weights_list])  # (G, n_in)
    x1_gain = np.stack([w["x1_gain"] for w in weights_list])  # (G, n_in)
    x1_ymin = np.array([w["x1_ymin"] for w in weights_list])  # (G,)

    # mapminmax_apply, batched: (x - xoffset) * gain + ymin, broadcasting each
    # net's own (n_in,) normalization against its own (n_in, Q) predictor slab.
    a = (x_stack - x1_xoffset[:, :, None]) * x1_gain[:, :, None] + x1_ymin[
        :, None, None
    ]

    for layer_index in range(n_layers):
        w_stack = np.stack(
            [w["layers"][layer_index][0] for w in weights_list]
        )  # (G, n_out, n_in)
        b_stack = np.stack(
            [w["layers"][layer_index][1] for w in weights_list]
        )  # (G, n_out)
        # Batched matrix-vector product: for each of the G nets independently,
        # W (n_out, n_in) @ a (n_in, Q) -> (n_out, Q). np.matmul (not
        # np.einsum!) is what actually dispatches this to BLAS -- einsum's
        # generic evaluator does NOT recognize this batched-contraction pattern
        # as a matmul, and measured 4-8x slower than np.matmul for these exact
        # shapes despite being mathematically identical (see run_nets module
        # docstring for the benchmark). np.matmul's batching semantics already
        # do exactly what the einsum subscripts spelled out: leading axes (here,
        # "g") are batch dimensions, the trailing two are the matrix pair.
        a = np.matmul(w_stack, a) + b_stack[:, :, None]
        if layer_index < n_layers - 1:
            a = _tansig(a)
    # The output layer always has exactly 1 unit (ESPER estimates one scalar
    # property per net) -- drop that axis.
    a = a[:, 0, :]  # (G, Q)

    y1_xoffset = np.array([w["y1_xoffset"] for w in weights_list])
    y1_gain = np.array([w["y1_gain"] for w in weights_list])
    y1_ymin = np.array([w["y1_ymin"] for w in weights_list])
    # mapminmax_reverse, batched: (y - ymin) / gain + xoffset, per net.
    return (a - y1_ymin[:, None]) / y1_gain[:, None] + y1_xoffset[:, None]


def run_nets(DesiredVariables, Equations, code={}):
    """
    Running neural nets

    Inputs:
        DesiredVariables: List of variables for estimates
        Equations: List of desired equations
        code: Dictionary of preprocessed measurements

    Outputs:
        EstAtl: Dictionary of estimates for the Atlantic and Arctic
            Oceans
        EstOther: Dictionary of estimates for not Altnatic/Arctic

    Implementation notes (batched rewrite)
    ---------------------------------------
    Every ``ESPER_{variable}_{equation}_{Atl,Other}_{1..4}`` net is a MATLAB
    ``genFunction()`` export: a small (2- or 3-layer) feedforward net, weights
    hard-coded as module-local literals. Profiling at production point counts
    (millions of points per chunk) found this loop's ORIGINAL per-net call --
    import one module, run its ``PyESPER_NN(X)`` -- spending the large majority
    of its time in each net's own ``mapminmax``/``tansig`` numpy calls, simply
    because there were up to ``len(DesiredVariables) * len(Equations) * 8``
    (variables x equations x 2 regions x 4 ensemble members) of them, each
    redoing the same shape of work over the *full* point count independently.
    (``.tolist()``/list-conversion overhead, initially the suspected culprit,
    measured to well under 1% of total time at 15M points -- not the problem.)

    The fix exploits a genuine, verified structural fact about every ESPER net
    archive: for a given equation, ensemble member ``n`` (1-4) has the *same*
    architecture (layer count and every layer's shape) for every target
    variable and both regions -- only the weight *values* differ. So instead of
    up to 48 separate per-net calls (6 variables x 2 regions x 4 members, for a
    single equation), the 2 regions x however-many-variables sharing member
    ``n``'s architecture are gathered into ONE call to :func:`batched_forward`
    per member per equation -- e.g. 4 batched calls total for a full 6-variable,
    single-equation ``run_nets`` invocation, each vectorized across every net
    that shares that member's shape.

    Grouping into 4 batched calls only accounts for part of the win, though --
    the other two pieces, found by actually profiling the batched version once
    it existed rather than assuming the job was done:

    * ``batched_forward``'s per-layer matmul originally used ``np.einsum``
      (a direct translation of the batched-matmul math). Measured 4-8x
      *slower* than the mathematically-identical ``np.matmul`` at these exact
      shapes -- ``einsum``'s generic evaluator does not recognize this
      contraction pattern as a batched matmul and does not dispatch to BLAS
      the way ``matmul`` does. Switched; see ``batched_forward``'s matmul line.
    * ``tansig`` (applied to every hidden layer's output) dominated the
      *remaining* time -- profiling the matmul-fixed version at 1M points
      found it responsible for over two-thirds of total runtime. Rewritten as
      a parallel numba kernel (``_tansig_kernel``); see ``_tansig``'s
      docstring for the breakdown and why the "obvious" ``np.tanh``
      alternative is actually slower here.

    Combined, measured end-to-end speedup over the original (pre-rewrite,
    still available at ``PyESPER/tests/_legacy_run_nets_reference.py``)
    implementation, all 6 ROMS/MARBL variables + 1 equation per call (the real
    cstar-forge/roms-tools usage pattern):

    ==========  =========  =========  =========
    n_points    original   this impl  speedup
    ==========  =========  =========  =========
    10,000      0.65s      0.38s      1.7x
    100,000     5.7s       0.83s      6.9x
    1,000,000   55.2s      6.9s       8.0x
    15,000,000  (untested -- see below)  161s   --
    ==========  =========  =========  =========

    (The original implementation was not re-run at 15M points -- at its own
    measured scaling it would take on the order of 15-20 minutes for this call
    alone, and that number is not the point; the point is 161 seconds is not
    remotely competitive with it either way.)

    The public contract (inputs, ``EstAtl``/``EstOther`` output shape) is
    unchanged -- this only changes how the numbers inside get computed.

    A note for whoever reviews this for upstream: the biggest remaining risk
    is thread oversubscription, not correctness. ``_tansig``'s numba kernel
    parallelizes internally across every core it can see; running multiple
    chunks of a large grid through ``run_nets`` concurrently (e.g. one
    dask/thread-pool worker per chunk) means each concurrent call fights the
    others for the same cores instead of one call getting to use all of them
    -- see ``_tansig``'s docstring for the full explanation and the mitigation
    (run chunks serially, one at a time, so this function's own parallelism is
    the only parallelism in play for that step).
    """

    # Predefining dictionaries to populate
    EstAtl, EstOther = {}, {}
    P, Sd, Td, Ad, Bd, Cd = {}, {}, {}, {}, {}, {}

    # Longitude/Latitude/Depth are the same shared metadata arrays for every
    # (DesiredVariable, Equation) combination in `code` (see iterations.py), so
    # cos/sin/tolist conversion is done once here instead of once per combination
    # -- previously redundant, repeated work that scaled with len(DesiredVariables).
    cosd = sind = lat = depth = None

    # Calculating inputs for nets and formatting them
    for name, value in code.items():
        if cosd is None:
            cosd = np.cos(np.deg2rad(value["Longitude"] - 20)).tolist()
            sind = np.sin(np.deg2rad(value["Longitude"] - 20)).tolist()
            lat, depth = value["Latitude"].tolist(), value["Depth"].tolist()
        # Convert columns to lists of floats
        Sd[name] = value["S"].astype(float).tolist()
        Td[name] = value["T"].astype(float).tolist()
        Ad[name] = value["A"].astype(float).tolist()
        Bd[name] = value["B"].astype(float).tolist()
        Cd[name] = value["C"].astype(float).tolist()

    # Define a mapping from equations to the list of variable dictionaries
    equation_map = {
        1: [Sd, Td, Ad, Bd, Cd],
        2: [Sd, Td, Ad, Cd],
        3: [Sd, Td, Bd, Cd],
        4: [Sd, Td, Cd],
        5: [Sd, Td, Ad, Bd],
        6: [Sd, Td, Ad],
        7: [Sd, Td, Bd],
        8: [Sd, Td],
        9: [Sd, Ad, Bd, Cd],
        10: [Sd, Ad, Cd],
        11: [Sd, Bd, Cd],
        12: [Sd, Cd],
        13: [Sd, Ad, Bd],
        14: [Sd, Ad],
        15: [Sd, Bd],
        16: [Sd],
    }

    # Ensemble members are always numbered 1-4 in the NeuralNetworks archive
    # (see the module docstring above for why member number alone determines
    # architecture, independent of variable/region/equation).
    members = (1, 2, 3, 4)
    regions = ("Atl", "Other")

    for e in Equations:
        # Build each requested variable's own (name, predictor array) once per
        # equation. Each array has shape (n_in, Q): the geographic columns shared
        # by every variable, followed by whichever of S/T/A/B/C this equation
        # uses -- but note the *values* behind "S"/"T"/etc are variable-specific
        # (see NeededForProperty/VarVec in iterations.py), so two variables' X
        # arrays are genuinely different even though every variable uses the same
        # equation's column layout.
        by_var = []
        for v in DesiredVariables:
            name = v + str(e)
            variables = [var[name] for var in equation_map[e]]
            x_arr = np.asarray(
                [cosd, sind, lat, depth] + variables, dtype=np.float64
            )  # (n_in, Q)
            by_var.append((v, name, x_arr))

        # Group by ensemble-member number: every (variable, region) pair sharing
        # member `n` has an identical architecture for this equation, so they can
        # all go through one batched_forward call together.
        for member in members:
            group_weights = []
            group_x = []
            group_slots = []  # (name, region) in the same order as the above two
            for v, name, x_arr in by_var:
                for region in regions:
                    module_name = f"NeuralNetworks.ESPER_{v}_{e}_{region}_{member}"
                    weights = parse_net_weights(module_name)
                    group_weights.append(weights)
                    group_x.append(x_arr)
                    group_slots.append((name, region))

            # Defensive, not load-bearing in practice: every module for a given
            # equation shares one architecture per member (verified across the
            # full NeuralNetworks archive -- see net_weights.architecture_signature
            # and the accompanying test), so this loop always has exactly one
            # group. If a future archive update ever broke that invariant, this
            # still computes the right answer -- just via more batched_forward
            # calls (one per distinct shape) instead of the fast path.
            by_signature = {}
            for weights, x_arr, slot in zip(group_weights, group_x, group_slots):
                by_signature.setdefault(architecture_signature(weights), []).append(
                    (weights, x_arr, slot)
                )

            for entries in by_signature.values():
                weights_list = [w for w, _x, _slot in entries]
                x_stack = np.stack([x for _w, x, _slot in entries])  # (G, n_in, Q)
                results = batched_forward(weights_list, x_stack)  # (G, Q)
                for row, (_w, _x, (name, region)) in zip(results, entries):
                    P.setdefault(name, {}).setdefault(region, {})[member] = row

    # Assemble into the original EstAtl/EstOther[name] = (n_points, 4) contract:
    # stack the 4 ensemble members' estimates for each (variable, equation) name,
    # per region -- identical to the original's
    # `np.stack([netstimateAtl[na][0] for na in range(4)], axis=1)`.
    for name, by_region in P.items():
        EstAtl[name] = np.stack([by_region["Atl"][m] for m in members], axis=1)
        EstOther[name] = np.stack([by_region["Other"][m] for m in members], axis=1)

    return EstAtl, EstOther
