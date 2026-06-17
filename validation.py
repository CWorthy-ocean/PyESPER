# %% Validating against GLODAP and comparing with MATLAB ESPERs too

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
import glodap
import pandas as pd
import numpy as np
from scipy.io import savemat

# Create the data
data = glodap.world() # version 2.2023
L = slice(0, 1000) # first 1000 data points

# Create the dictionary of predictors
PredictorMeasurements = {
    k: data[k][L].values.tolist()
    for k in [
        "salinity",
        "temperature",
        "phosphate",
        "nitrate",
        "silicate",
        "oxygen",
    ]
}

# Create the dictionary of coordinates
OutputCoordinates = {
    k: data[k][L].values.tolist()
    for k in [
        "longitude",
        "latitude",
        "depth",
    ]
}
# Create the list of estimation dates, accounting for leap years and YYYYMMDD
dates = pd.to_datetime(data["datetime"][L].values)

EstDates = (
    dates.year
    + (dates.dayofyear - 1)
    / np.where(dates.is_leap_year, 366, 365)
).tolist()

# Define the path
Path = ""  # home directory
# %% Now try PyESPERs LIR
             
EstimatesLIR, CoefficientsLIR, UncertaintiesLIR = lir(
    ['oxygen'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates
    )

# DEBUG, unhash as needed
print('LIR Estimates are:')
print(EstimatesLIR['oxygen1'][700:710]) 
print('LIR Coefficient A estimates are:')
print(CoefficientsLIR['oxygen1']['Coef A'][5:10])
print('LIR uncertainties are:')
print(UncertaintiesLIR['oxygen1'][308:310])
# %% NN

EstimatesNN, UncertaintiesNN = nn(
    ['oxygen'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates
    )

# DEBUG, unhash as needed
print('NN Estimates are:')
print(EstimatesNN['oxygen10'][300:310]) 
print('NN uncertainties are:')
print(UncertaintiesNN['oxygen10'][508:510])
# %% Mixed

EstimatesMixed, UncertaintiesMixed = mixed(
    ['oxygen'], 
    Path,
    OutputCoordinates, 
    PredictorMeasurements,
    EstDates=EstDates
    )

# DEBUG, unhash as needed
print('Mixed Estimates are:')
print(EstimatesMixed['oxygen3'][980:990]) 
print('Mixed uncertainties are:')
print(UncertaintiesMixed['oxygen3'][712:715])
# %% Save

savemat(
    "validation_review_main_oxygen.mat",
    {
        "Estimates_LIR": EstimatesLIR,
        "Estimates_NN": EstimatesNN,
        "Estimates_Mixed": EstimatesMixed,
        "Lon": OutputCoordinates["longitude"],
        "Lat": OutputCoordinates["latitude"],
        "Depth": OutputCoordinates["depth"],
    },
)
