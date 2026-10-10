def _build_uncertainty_interpolant(Path, variable):
    """Build the RMSE interpolant for one variable (cached by ``grid_cache``).

    Reproduces exactly what ``scipy.interpolate.griddata(..., method="linear")`` does
    internally -- ``LinearNDInterpolator(column_stack(points), values)`` with griddata's
    own defaults (``fill_value=nan``, ``rescale=False``) -- so calling the returned
    object is bit-identical to the griddata call it replaces. The only change is that
    the triangulation is built once instead of once per variable per equation per call.
    """
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator

    from PyESPER.fetch_polys_NN import fetch_polys_NN

    NN_data = fetch_polys_NN(Path, [variable])

    # The 1408 nodes are stored as a nested 16 (equation) x 11 (salinity) x 8 (depth)
    # structure; flatten in the original a-major order so the node ordering -- and hence
    # the triangulation -- is unchanged.
    grid = NN_data[1]
    data_arrays = [
        np.nan_to_num(
            np.array(
                [
                    grid[i][c][b][a]
                    for a in range(16)
                    for b in range(11)
                    for c in range(8)
                ]
            )
        )
        for i in range(4)
    ]
    u_depth, u_sal, eqn, rmse = data_arrays

    points = np.column_stack((u_depth, u_sal, eqn))
    return LinearNDInterpolator(points, rmse)


def emlr_nn(Path, DesiredVariables, Equations, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):

    """
    Estimating EMLR for neural networks.
    Returns a dictionary with (DesiredVariable, Equation) as keys and Uncertainties as values.
    """

    import numpy as np

    from PyESPER.kernels import grid_cache

    EMLR = {}

    depth = np.asarray(OutputCoordinates["depth"])
    salinity = np.asarray(PredictorMeasurements["salinity"])

    for dv in DesiredVariables:
        DV = f"{dv}"
        interpolant = grid_cache.nn_uncertainty_interpolant(
            Path, DV, lambda variable=DV: _build_uncertainty_interpolant(Path, variable)
        )

        for eq in Equations:
            name = dv + str(eq)
            eq_array = np.full_like(depth, eq, dtype=float)

            # Perform estimation for each equation
            EMLR[name] = interpolant(np.column_stack((depth, salinity, eq_array)))

    return EMLR
