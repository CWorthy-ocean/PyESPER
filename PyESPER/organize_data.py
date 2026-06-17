def organize_data(
    aaLCs,
    elLCs,
    aaInterpolants_pre,
    elInterpolants_pre,
    Gdf={},
    AAdata={},
    Elsedata={},
):
    """
    Organize interpolation output into more usable formatting and compute estimates
    Inputs:
        aaLCs: List of coefficients for input data from Atlantic/Arctic regions
        elLCs: List of coefficients for input data not from Atlantic/Arctic
        aaInterpolants_pre: Scipy interpolant for Atlantic/Arctic
        elInterpolants_pre: Scipy interpolant for not Atlantic/Arctic
        Gdf: Dictionary of grid for interpolation, separated into regions
        AAdata: Dictionary of user input data for Atlantic/Arctic
        ElseData: Dictionary of user input data not for Atlantic/Arctic
    Outputs:
        Estimate: Dictionary of estimates for each equation-desired variable combination
        CoefficientsUsed: Dictionary of dictionaries of coefficients for each equation-
            desired variable combination
    """
    import numpy as np
    import re

    Estimate = {}
    CoefficientsUsed = {}
    Gkeys = list(Gdf.keys())

    def safe_float(arr):  # nix a persistant bad string
        array_object = np.asarray(arr)
        if array_object.dtype.kind in {"U", "S", "O"}:
            array_object = np.where(
                array_object == "-2.4319000000000003e-",
                "-0.000002",
                array_object,
            )
        try:
            return array_object.astype(np.float64)
        except (ValueError, TypeError):

            def conv(x):
                try:
                    return float(x)
                except:
                    return np.nan

            return np.vectorize(conv, otypes=[float])(array_object)

    def process_pair(c_in, v_in):  # make sure NaNs are treated like MATLAB
        c_out = np.asarray(c_in, dtype=np.float64)
        v_out = safe_float(v_in)
        return c_out, v_out

    for i, key in enumerate(Gkeys):
        eq_num = int(re.search(r"(\d+)$", key).group(1))
        mask = 16 - eq_num  # this was backwards previously

        do_B = (mask & 1) != 0
        do_A = (mask & 2) != 0
        do_C = (mask & 4) != 0
        do_T = (mask & 8) != 0

        for region_data, region_lcs in [
            (AAdata[key], aaLCs[i]),
            (Elsedata[key], elLCs[i]),
        ]:
            if len(region_data["Order"]) == 0:
                continue

            c_alpha = region_lcs[:, 0]
            c_S = region_lcs[:, 1]

            v_S = region_data["S"]

            arr_len = len(v_S)

            if do_T:
                c_T, v_T = process_pair(region_lcs[:, 2], region_data["T"])
            else:
                c_T = np.zeros(arr_len, dtype=np.float64)
                v_T = np.zeros(arr_len, dtype=np.float64)

            if do_A:
                c_A, v_A = process_pair(region_lcs[:, 3], region_data["A"])
            else:
                c_A = np.zeros(arr_len, dtype=np.float64)
                v_A = np.zeros(arr_len, dtype=np.float64)

            if do_B:
                c_B, v_B = process_pair(region_lcs[:, 4], region_data["B"])
            else:
                c_B = np.zeros(arr_len, dtype=np.float64)
                v_B = np.zeros(arr_len, dtype=np.float64)

            if do_C:
                c_C, v_C = process_pair(region_lcs[:, 5], region_data["C"])
            else:
                c_C = np.zeros(arr_len, dtype=np.float64)
                v_C = np.zeros(arr_len, dtype=np.float64)

            c_alpha = np.asarray(c_alpha, dtype=np.float64)
            c_S = np.asarray(c_S, dtype=np.float64)
            v_S = safe_float(v_S)

            est = (
                c_alpha
                + c_S * v_S
                + c_T * v_T
                + c_A * v_A
                + c_B * v_B
                + c_C * v_C
            )

            region_data["Estimate"] = est
            region_data["C0"] = c_alpha
            region_data["CS"] = c_S
            region_data["CT"] = c_T
            region_data["CA"] = c_A
            region_data["CB"] = c_B
            region_data["CC"] = c_C

        merged = {}
        aa_d = AAdata[key]
        el_d = Elsedata[key]
        for k in ["Estimate", "C0", "CS", "CT", "CA", "CB", "CC", "Order"]:
            aa_val = aa_d.get(k, np.array([]))
            el_val = el_d.get(k, np.array([]))
            merged[k] = np.concatenate([aa_val, el_val])

        idx = np.argsort(merged["Order"])
        Estimate[key] = merged["Estimate"][idx]
        CoefficientsUsed[key] = {
            "Intercept": merged["C0"][idx],
            "Coef S": merged["CS"][idx],
            "Coef T": merged["CT"][idx],
            "Coef A": merged["CA"][idx],
            "Coef B": merged["CB"][idx],
            "Coef C": merged["CC"][idx],
        }

    return Estimate, CoefficientsUsed
