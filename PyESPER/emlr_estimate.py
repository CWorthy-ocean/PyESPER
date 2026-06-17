import numpy as np
from scipy.interpolate import LinearNDInterpolator
from numba import njit, prange


@njit(parallel=True, cache=True)
def calculate_uncertainty_kernel(
    s_raw,
    t_raw,
    a_raw,
    b_raw,
    c_raw,
    u_s_raw,
    u_t_raw,
    u_a_raw,
    u_b_raw,
    u_c_raw,
    du_s_raw,
    du_t_raw,
    du_a_raw,
    du_b_raw,
    du_c_raw,
    rmse_interpolated,
    eq_val,
):
    """
    Numba JIT compiled uncertainty propagation.
    """

    number_of_points = s_raw.shape[0]
    output_uncertainty = np.full(number_of_points, np.nan, dtype=np.float64)

    req_t = eq_val in (1, 2, 3, 4, 5, 6, 7, 8)
    req_a = eq_val in (1, 2, 5, 6, 9, 10, 13, 14)
    req_b = eq_val in (1, 3, 5, 7, 9, 11, 13, 15)
    req_c = eq_val in (1, 2, 3, 4, 9, 10, 11, 12)

    for i in prange(number_of_points):
        u_s = -9999.0 if np.isnan(u_s_raw[i]) else u_s_raw[i]
        u_t = -9999.0 if np.isnan(u_t_raw[i]) else u_t_raw[i]
        u_a = -9999.0 if np.isnan(u_a_raw[i]) else u_a_raw[i]
        u_b = -9999.0 if np.isnan(u_b_raw[i]) else u_b_raw[i]
        u_c = -9999.0 if np.isnan(u_c_raw[i]) else u_c_raw[i]

        mask_val = u_s == -9999.0
        if req_t:
            mask_val = mask_val or (u_t == -9999.0)
        if req_a:
            mask_val = mask_val or (u_a == -9999.0)
        if req_b:
            mask_val = mask_val or (u_b == -9999.0)
        if req_c:
            mask_val = mask_val or (u_c == -9999.0)

        if mask_val:
            output_uncertainty[i] = np.nan
            continue

        du_s = -9999.0 if np.isnan(du_s_raw[i]) else du_s_raw[i]
        du_t = -9999.0 if np.isnan(du_t_raw[i]) else du_t_raw[i]
        du_a = -9999.0 if np.isnan(du_a_raw[i]) else du_a_raw[i]
        du_b = -9999.0 if np.isnan(du_b_raw[i]) else du_b_raw[i]
        du_c = -9999.0 if np.isnan(du_c_raw[i]) else du_c_raw[i]

        s = 0.0 if np.isnan(s_raw[i]) else s_raw[i]
        t = 0.0 if np.isnan(t_raw[i]) else t_raw[i]
        a = 0.0 if np.isnan(a_raw[i]) else a_raw[i]
        b = 0.0 if np.isnan(b_raw[i]) else b_raw[i]
        c = 0.0 if np.isnan(c_raw[i]) else c_raw[i]

        sum_squared = (
            (s * u_s) ** 2
            + (t * u_t) ** 2
            + (a * u_a) ** 2
            + (b * u_b) ** 2
            + (c * u_c) ** 2
        )

        delta_sum_squared = (
            (s * du_s) ** 2
            + (t * du_t) ** 2
            + (a * du_a) ** 2
            + (b * du_b) ** 2
            + (c * du_c) ** 2
        )

        variance_result = (
            sum_squared - delta_sum_squared + rmse_interpolated[i] ** 2
        )

        if np.isnan(variance_result):
            output_uncertainty[i] = np.nan
        elif variance_result >= 0:
            output_uncertainty[i] = np.sqrt(variance_result)
        else:
            output_uncertainty[i] = 0.0

    return output_uncertainty


def emlr_estimate(
    Equations,
    DesiredVariables,
    Path,
    OutputCoordinates={},
    PredictorMeasurements={},
    UDict={},
    DUDict={},
    Coefficients={},
    verbose=False,
    **kwargs,
):
    """
    Uncertainty estimation step 1 for LIRs
    Inputs:
        Equations: List of equations
        DesiredVariables: List of variables to estimate
        Path: User-defined computer path
        OutputCoordinates: Dictionary of locations where estimates are requested
        PredictorMeasurements: Dictionary of measurements provided by user
        UDict: Dictionary of user-defined measurement uncertainties
        DUDict: Dictionary of default measurement uncertainties
        Coefficients: Dictionary of dictionaries of coefficients for each
            variable-equation scenario
        **kwargs: Please see README for full description
    Output:
        EMLR: Dictionary of uncertainty values for each desired variable-equation
            case scenario and estimate
    """
    from PyESPER.fetch_data import fetch_data

    if verbose:
        print("Propagating uncertainties.")
    EMLR = {}
    depth_out = np.asarray(OutputCoordinates["depth"], dtype=np.float64)
    sal_out = np.asarray(PredictorMeasurements["salinity"], dtype=np.float64)
    query_points_2d = np.column_stack((depth_out, sal_out))

    for dv in DesiredVariables:
        LIR_data = fetch_data([dv], Path)

        arr = np.array(LIR_data)
        arr = arr[3]
        arritem = arr.item()

        raw_list = [
            [
                arritem[i][c][b][a]
                for a in range(16)
                for b in range(11)
                for c in range(8)
            ]
            for i in range(len(arritem))
        ]

        UGridArray = np.array(raw_list, dtype=np.float64)
        np.nan_to_num(UGridArray, copy=False)
        UGridArray = UGridArray.T

        UDepth, USal, Eqn, RMSE = UGridArray.T

        unique_equations = np.unique(Eqn)
        number_of_equations = len(unique_equations)

        base_mask = Eqn == unique_equations[0]
        points_2d = np.column_stack((UDepth[base_mask], USal[base_mask]))

        rmse_matrix = np.empty(
            (points_2d.shape[0], number_of_equations), dtype=np.float64
        )
        for i, eq_val in enumerate(unique_equations):
            rmse_matrix[:, i] = RMSE[Eqn == eq_val]

        interpolator = LinearNDInterpolator(points_2d, rmse_matrix)
        all_rmse_interpolated = interpolator(query_points_2d)

        for eq in Equations:
            combo = f"{dv}{eq}"

            if combo not in UDict or eq not in unique_equations:
                continue

            equation_index = np.where(unique_equations == eq)[0][0]
            rmse_interpolated = all_rmse_interpolated[:, equation_index]

            uncdfs = UDict[combo]
            duncdfs = DUDict[combo]
            keys = list(uncdfs.keys())

            u_s_raw = np.asarray(uncdfs[keys[0]], dtype=np.float64)
            u_t_raw = np.asarray(uncdfs[keys[1]], dtype=np.float64)
            u_a_raw = np.asarray(uncdfs[keys[2]], dtype=np.float64)
            u_b_raw = np.asarray(uncdfs[keys[3]], dtype=np.float64)
            u_c_raw = np.asarray(uncdfs[keys[4]], dtype=np.float64)

            du_s_raw = np.asarray(duncdfs[keys[0]], dtype=np.float64)
            du_t_raw = np.asarray(duncdfs[keys[1]], dtype=np.float64)
            du_a_raw = np.asarray(duncdfs[keys[2]], dtype=np.float64)
            du_b_raw = np.asarray(duncdfs[keys[3]], dtype=np.float64)
            du_c_raw = np.asarray(duncdfs[keys[4]], dtype=np.float64)

            s_raw = np.asarray(UDict[combo]["US"], dtype=np.float64)
            t_raw = np.asarray(UDict[combo]["UT"], dtype=np.float64)
            a_raw = np.asarray(UDict[combo]["UA"], dtype=np.float64)
            b_raw = np.asarray(UDict[combo]["UB"], dtype=np.float64)
            c_raw = np.asarray(UDict[combo]["UC"], dtype=np.float64)

            final_uncertainty = calculate_uncertainty_kernel(
                s_raw,
                t_raw,
                a_raw,
                b_raw,
                c_raw,
                u_s_raw,
                u_t_raw,
                u_a_raw,
                u_b_raw,
                u_c_raw,
                du_s_raw,
                du_t_raw,
                du_a_raw,
                du_b_raw,
                du_c_raw,
                rmse_interpolated,
                int(eq),
            )

            EMLR[combo] = final_uncertainty

    return EMLR
