import functools


@functools.lru_cache(maxsize=None)
def _load_net(module_name):
    """Import a hard-coded ESPER neural-net weight module once and return its
    ``PyESPER_NN`` forward-pass function.

    The ``NeuralNetworks.ESPER_*`` modules are static (hard-coded weight matrices),
    so importing them once per process and caching is both correct and much faster
    than the original per-call ``importlib.reload``. Removing ``reload`` also makes
    this path thread-safe (``importlib.reload`` mutates ``sys.modules`` and is not),
    which matters when the estimate is driven per dask chunk.
    """
    import importlib

    return importlib.import_module(module_name).PyESPER_NN


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
    """

    import numpy as np

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
        16: [Sd]
    }

    # Create the correct vector for each equation case
    for e in Equations:
        for v in DesiredVariables:
            name = v + str(e)
            # Get the corresponding variables for the equation
            variables = [var[name] for var in equation_map[e]]
            P[name] = [[[cosd, sind, lat, depth] + variables]]
            netstimateAtl, netstimateOther = [], []
            for n in range(1, 5):
                fOName = f"NeuralNetworks.ESPER_{v}_{e}_Other_{n}"
                fAName = f"NeuralNetworks.ESPER_{v}_{e}_Atl_{n}"
                # Cached, reload-free import (see _load_net): thread-safe and avoids
                # re-importing the static weight modules on every call/chunk.
                net_atl = _load_net(fAName)
                net_other = _load_net(fOName)
                # Running the nets
                netstimateAtl.append(net_atl(P[name]))
                netstimateOther.append(net_other(P[name]))

            # Process estimates for Atlantic and Other regions. Each net output has
            # shape (1, n_points); stack the 4 ensemble members into an (n_points, 4)
            # array. This replaces a per-point double Python list comprehension
            # (O(n_points) Python-level iterations) with a single vectorized numpy
            # call -- previously the dominant cost at large n_points.
            EstAtlL = np.stack([netstimateAtl[na][0] for na in range(4)], axis=1)
            EstOtherL = np.stack([netstimateOther[no][0] for no in range(4)], axis=1)

            # Store the result
            EstAtl[name] = EstAtlL
            EstOther[name] = EstOtherL
            
    return EstAtl, EstOther

