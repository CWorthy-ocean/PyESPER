def process_netresults(Equations, code={}, df={}, EstAtl={}, EstOther={}):

    """
    Regional smoothing and processing net outputs

    Inputs:
        Equations: List of equations to use
        code: Dictionary of input for each equation-variable scenario
        df: Dictionary of coordinates and boolean indicators for regions
        EstAtl: Dictionary of estimates for the Atlantic and Arctic
            for each combination
        EstOther: Dictionary of estimates for not Atlantic and Arctic,
            for each combination

    Output:
        Estimate: Dictionary of estimates for each combination

    Notes
    -----
    This is a vectorized rewrite of the original per-point Python loops. Every
    ``np.tile(...)`` in the original smoothing loops tiled a *scalar* to shape
    ``(1, len(Equations))`` and then immediately re-extracted index ``[0][0]`` --
    i.e. the tile width never affected the result, so the tiling was pure
    overhead. Replacing the per-point loops with whole-array numpy expressions
    is therefore numerically identical to the original, just without the
    O(n_points) Python-level `np.tile`/`np.mean` call overhead that dominated
    runtime at large n_points.
    """

    import numpy as np

    # Function to provide the mean of values when needed. `estimates[key]` is
    # an (n_points, 4) array (stacked ensemble members from run_nets); a single
    # vectorized mean over axis=1 replaces the previous per-point `np.mean()`
    # call in a Python loop.
    def process_estimates(estimates):
        return {
            key: np.asarray(value, dtype=float).mean(axis=1)
            for key, value in estimates.items()
        }

    Esta = process_estimates(EstAtl)
    Esto = process_estimates(EstOther)

    # Processing regionally in the Atlantic and Bering
    EstA, EstB, EB2, ESat, ESat2, ESaf, Estimate  = {}, {}, {}, {}, {}, {}, {}

    for i in code:
        code[i]["AAInds"] = df["AAInds"]
        code[i]["BeringInds"] = df["BeringInds"]
        code[i]["SAtlInds"] = df["SAtlInds"]
        code[i]["SoAfrInds"] = df["SoAfrInds"]

    for codename, codedata in code.items():
        aainds, beringinds, satlinds, latitude, safrinds = (
            codedata[key] for key in ["AAInds", "BeringInds", "SAtlInds", "Latitude", "SoAfrInds"]
        )
        latitude = np.asarray(latitude, dtype=float)
        aainds = np.asarray(aainds, dtype=bool)
        beringinds = np.asarray(beringinds, dtype=bool)
        # SAtlInds/SoAfrInds are lists of the strings 'True'/'False' (see
        # define_polygons), not booleans -- match the original string comparison.
        satlinds_bool = np.asarray(satlinds) == "True"
        safrinds_bool = np.asarray(safrinds) == "True"

        esta = Esta[codename]
        esto = Esto[codename]

        Estatl = np.where(aainds, esta, esto)

        # Smoothing for the Atlantic
        Estb = esta * ((latitude - 62.5) / 7.5) + esto * ((70.0 - latitude) / 7.5)

        eb2 = np.where(beringinds, Estb, Estatl)

        # Smoothing for the Bering
        Estsat = esta * ((latitude + 44.0) / 10.0) + esto * ((-34.0 - latitude) / 10.0)

        EstA[codename], EstB[codename], EB2[codename], ESat[codename] = Estatl, Estb, eb2, Estsat

        # Regional processing for S. Atlantic
        ESat2[codename] = np.where(satlinds_bool, ESat[codename], EB2[codename])

        # Regional processing for S. Africa
        lon = np.asarray(df["Lon"], dtype=float)
        esafr = ESat2[codename] * ((27.0 - lon) / 8.0) + esto * ((lon - 19.0) / 8.0)
        ESaf[codename] = esafr

        Estimate[codename] = np.where(safrinds_bool, ESaf[codename], ESat2[codename])

    # Bookkeeping blanks back to NaN as needed (dead in practice: v is now an
    # ndarray, which -- like the original list -- never compares equal to '').
    # Convert back to plain Python lists to preserve the original return-type
    # contract of this function (some callers besides roms-tools may rely on
    # list semantics, e.g. JSON serialization).
    Estimate = {
        k: ('NaN' if isinstance(v, str) and v == '' else v.tolist())
        for k, v in Estimate.items()
    }

    return Estimate
