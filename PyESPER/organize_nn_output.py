def organize_nn_output(Path, DesiredVariables, OutputCoordinates={}, PredictorMeasurements={},  **kwargs):

    """
    Iteratively calculate uncertainties

    Inputs:
        Path: User-defined computer path
        Desired Variables: List of variables requested
        OutputCoordinates: Dictionary of geographic coordinates
        PredictorMeasurements: Dictionary of predictor measurements
        **kwargs: Please see README

    Outputs:
        Uncertainties: Dictionary of uncertainty estimates for each
            equation-variable combination
    """

    import numpy as np
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.define_polygons import define_polygons
    from PyESPER.run_nets import run_nets
    from PyESPER.process_netresults import process_netresults
    from PyESPER.emlr_nn import emlr_nn

    # Predefine output lists
    PD_final, DPD_final, Unc_final, DUnc_final = [], [], [], []
    emlr = []

    # Rerun nets for each variable
    for d, var in enumerate(DesiredVariables):
        Pertdv, DPertdv, Unc, DUnc = [], [], [], []
        var = [var] # Wrap single variable in a list
        keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]

        Equations, n, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
            var,
            PredictorMeasurements,
            OutputCoordinates,
            **kwargs
        )
        Uncertainties_pre, DUncertainties_pre = measurement_uncertainty_defaults(
            n,
            PredictorMeasurements,
            MeasUncerts
        )
    
        InputAll = inputdata_organize( 
            EstDates,
            C,
            PredictorMeasurements,
            Uncertainties_pre
        )
    
        PredictorMeasurements, InputAll = temperature_define(
            var,
            PredictorMeasurements,
            InputAll,
            **kwargs
        )
            
        code, unc_combo_dict, dunc_combo_dict = iterations(
            var,
            Equations,
            PerKgSwTF,
            C,
            PredictorMeasurements,
            InputAll,
            Uncertainties_pre,
            DUncertainties_pre
        )

        df = define_polygons(C)
            
        EstAtl, EstOther = run_nets(var, Equations, code)
            
        Estimate = process_netresults(
            Equations,
            code,
            df,
            EstAtl,
            EstOther
        )
            
        EMLR = emlr_nn(
            Path,
            var,
            Equations,
            OutputCoordinates,
            PredictorMeasurements
        )
      
        emlr.append(EMLR)
        names = list(PredictorMeasurements.keys())

        # Predictors as one (n_points, n_predictors) float64 matrix.
        #
        # This was a per-point double list comprehension
        # (``[[0 if val == "nan" else val for val in pm] for pm in PMs]``) followed by
        # np.transpose. Note what that comparison actually did: ``val == "nan"`` tests
        # against the *string* "nan", so it never matched a float NaN -- for the numeric
        # input every caller in this package supplies it was a pure copy. The string
        # form is still honoured below, for a caller passing the documented list-of-
        # strings, so this is behaviour-preserving rather than a tightening.
        PMs = _predictor_matrix(PredictorMeasurements.values())
            
        PMs3, DMs3 = {}, {}
            
        for pred in range(len(PredictorMeasurements)):
            num_coords = len(OutputCoordinates["longitude"])
            num_preds = len(PredictorMeasurements)
            
            # Initialize perturbation arrays
            Pert = np.zeros((num_coords, num_preds))
            DefaultPert = np.zeros((num_coords, num_preds))
            
            # Populate perturbation arrays
            Pert[:, pred] = Uncertainties_pre[keys[pred]]
            DefaultPert[:, pred] = DUncertainties_pre[keys[pred]]
            
            # Apply perturbations
            PMs2 = PMs + Pert
            DMs2 = PMs + DefaultPert
        
            # Update PMs3 and DMs3 dictionaries.
            #
            # Contiguous float64 copies rather than ``.tolist()``: a list of n Python
            # floats costs ~40 bytes/point against 8, and every consumer downstream
            # (defaults/inputdata_organize/iterations/temperature_define) immediately
            # converts back to an array. Copies rather than views because
            # temperature_define may write into these in place, and a view would
            # reach back into PMs2/DMs2.
            for col, name in enumerate(names):
                PMs3[name] = np.ascontiguousarray(PMs2[:, col])
                DMs3[name] = np.ascontiguousarray(DMs2[:, col])
        
            # Run preprocess_applynets for perturbed and default data
            VTF = False
            Eqs2, n2, VerbTF2, EstDates2, C2, PerKgSwTF2, MeasUncerts2 = defaults(
                var,
                PMs2,
                OutputCoordinates,
                Equations=Equations,
                EstDates=EstDates,
                VerboseTF = VTF,
            )
            U_pre2, DU_pre2 = measurement_uncertainty_defaults(
                n2,
                PMs3,
                MeasUncerts2
            )
            InputAll2 = inputdata_organize(
                EstDates2,
                C2,
                PMs3,
                U_pre2
            )
            InputAll3 = inputdata_organize(   
                EstDates2,
                C2,
                DMs3,
                U_pre2
            )
            PMs3, InputAll2 = temperature_define(
                var,
                PMs3,
                InputAll2,
                **kwargs
            )
            DMs3, InputAll3 = temperature_define(
                var,
                DMs3,
                InputAll3,
                **kwargs
            )
            code2, unc_combo_dict2, dunc_combo_dict2 = iterations(
                var,
                Eqs2,
                PerKgSwTF2,
                C2,  
                PMs3, 
                InputAll2,
                U_pre2,
                DU_pre2   
            )
            code3, unc_combo_dict3, dunc_combo_dict3 = iterations(
                var,  
                Eqs2,
                PerKgSwTF2,
                C2, 
                DMs3,
                InputAll3,
                U_pre2, 
                DU_pre2
            )
            df2 = define_polygons(C2)
            EstAtl2, EstOther2 = run_nets(
                var,
                Eqs2,   
                code2
            )
            EstAtl3, EstOther3 = run_nets(
                var, 
                Eqs2,
                code3
            )
            PertEst = process_netresults(
                Eqs2,  
                code2,    
                df2,
                EstAtl2,
                EstOther2
            )
            DefaultPertEst = process_netresults(
                Eqs2,
                code3,
                df2,
                EstAtl3,
                EstOther3
            )
            
            # Extract estimates and perturbation results
            estimates = [np.array(v) for v in Estimate.values()]
            pertests = [np.array(v) for v in PertEst.values()]
            defaultpertests = [np.array(v) for v in DefaultPertEst.values()]
             
            # Initialize result lists
            PertDiff, DefaultPertDiff, Unc_sub2, DUnc_sub2 = [], [], [], []

            for c in range(len(Equations)):
            # Compute differences and squared differences using numpy
                PD = estimates[c] - pertests[c]
                DPD = estimates[c] - defaultpertests[c]
                Unc_sub1 = (estimates[c] - pertests[c])**2
                DUnc_sub1 = (estimates[c] - defaultpertests[c])**2
                
               # Append results
                PertDiff.append(PD)
                DefaultPertDiff.append(DPD)
                Unc_sub2.append(Unc_sub1)
                DUnc_sub2.append(DUnc_sub1)
            Pertdv.append(PertDiff)
            DPertdv.append(DefaultPertDiff)
            Unc.append(Unc_sub2)
            DUnc.append(DUnc_sub2)
        PD_final.append(Pertdv)
        DPD_final.append(DPertdv)
        Unc_final.append(Unc)
        DUnc_final.append(DUnc)
            
    # Compute final uncertainty propagation
            
    emlr_combined = {k: v for d in emlr for k, v in d.items()}
    n_predictors = len(PredictorMeasurements)
    Uncertainties = {}

    # Combine the perturbation contributions.
    #
    # This was a loop over every point, inside loops over variable and equation,
    # building two n_predictors-long np.arrays and calling np.sum on each -- at 100k
    # points and six variables, 1.2M array allocations and 1.2M ufunc reductions, and
    # half the total runtime of the uncertainty path. Summing along a stacked
    # predictor axis instead is the identical arithmetic: the same float64 addends
    # accumulated in the same order (numpy reduces axis 0 sequentially), so the result
    # is bit-identical -- pinned by test_uncertainty.py.
    #
    # np.sqrt of a negative variance still yields NaN, exactly as before; unlike the
    # LIR kernel this path deliberately does not clamp to zero.
    for dv, variable in enumerate(DesiredVariables):
        for eq_index, equation in enumerate(Equations):
            name = f"{variable}{equation}"
            u = np.stack([
                np.asarray(Unc_final[dv][pre][eq_index], dtype=np.float64).ravel()
                for pre in range(n_predictors)
            ])
            du = np.stack([
                np.asarray(DUnc_final[dv][pre][eq_index], dtype=np.float64).ravel()
                for pre in range(n_predictors)
            ])
            eu = np.asarray(emlr_combined[name], dtype=np.float64).ravel()
            Uncertainties[name] = np.sqrt(u.sum(axis=0) - du.sum(axis=0) + eu**2)

    return Uncertainties


def _predictor_matrix(values):
    """Stack predictor columns into one ``(n_points, n_predictors)`` float64 array.

    Accepts the numeric arrays the package uses internally and also the
    list-of-strings form the public API documents, where a missing value is the
    literal ``"nan"`` and the original code mapped it to 0.
    """
    import numpy as np

    columns = []
    for value in values:
        column = np.asarray(value)
        if column.dtype.kind in "OUS":
            # Legacy string input: "nan" meant zero here, not NaN.
            column = np.where(column == "nan", 0.0, column).astype(np.float64)
        else:
            column = column.astype(np.float64, copy=False)
        columns.append(column)
    return np.column_stack(columns)
