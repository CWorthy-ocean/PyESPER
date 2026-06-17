def adjust_pH_DIC(
    DesiredVariables,
    verbose,
    Dates,
    Path,
    Est_pre={},
    PredictorMeasurements={},
    OutputCoordinates={},
    **kwargs
):
    """
    If present, adjusting pH and DIC for anthropogenic carbon (Cant) within LIRs. Cant adjustment methods
        are based on those from ESPERv1, which is a TTD-based assumption/simplification but does not
        use the Tracer-based Rapid Anthropogenic Carbon Estimation data product, TRACE. Rather,
        interpolation from a gridded product is used to produce estimates for the year 2002 and data is
        adjusted to/from this reference year. This is the first of three steps for Cant adjustment

    Inputs:
        DesiredVariables: List of desired variables to estimate
        verbose: Boolean indicating whether the user wants suppression of warnings
        Dates: List of dates for estimates
        Est_pre: Dictionary of preliminary estimates for each variable-equation case scenario
        PredictorMeasurements: Dictionary of input measurements for each variable-equation case scenario
        OutputCoordinates: Dictionary of coordinates for locations of estimates
        **kwargs: Please see README for full informations

    Outputs:
        Cant_adjusted: Dictionary of values adjusted for anthropogenic carbon for each combination
        Cant: Numpy array of estimates for anthropogenic carbon for each estimate
        Cant2002: Numpy array of estimates for anthropogenic carbon in the year 2002 for each estimate
    """

    import numpy as np
    from PyESPER.simplecantestimatelr import simplecantestimatelr

    if "DIC" not in DesiredVariables and "pH" not in DesiredVariables:
        n = len(Dates) if hasattr(Dates, "__len__") else 1
        return Est_pre, np.zeros(n), np.zeros(n)

    if verbose:
        print("Estimating anthropogenic carbon for PyESPER_LIR.")

    longitude = np.mod(np.asarray(OutputCoordinates["longitude"]), 360.0)
    latitude = np.asarray(OutputCoordinates["latitude"])
    depth = np.asarray(OutputCoordinates["depth"])

    Cant, Cant2002 = simplecantestimatelr(
        Dates, longitude, latitude, depth, Path
    )
    Cant = np.asarray(Cant, dtype=np.float64)
    Cant2002 = np.asarray(Cant2002, dtype=np.float64)
    cant_diff = Cant - Cant2002

    Cant_adjusted = {}

    for combo, val in Est_pre.items():
        val_arr = np.asarray(val, dtype=np.float64).flatten()
        combo_lower = combo.lower()

        if "dic" in combo_lower:
            adjusted = val_arr + cant_diff
            Cant_adjusted[combo] = adjusted
        else:
            Cant_adjusted[combo] = val_arr

    return Cant_adjusted, Cant, Cant2002
