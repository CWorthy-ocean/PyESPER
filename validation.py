# %% EXAMPLE

# Basic example first

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
import glodap

# Create the data
data = glodap.world()

L = slice(0, 10000) # Start with the first 10,000 measurements

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

# Create the list of estimation dates; change this later to decimal year
EstDates = data.year[L].values.tolist()

# Define the path
Path = "" 
             
# First trying 
EstimatesLIR, CoefficientsLIR, UncertaintiesLIR = lir(
    ['DIC'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates,
    Equations=[2]
    )

EstimatesNN, UncertaintiesNN = nn(
    ['phosphate'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates,
    Equations=[3]
    )

#EstimatesMixed, UncertaintiesMixed = mixed(
#    ['pH'], 
#    Path,
#    OutputCoordinates, 
#    PredictorMeasurements,
#    EstDates=EstDates,
#    Equations=[15]
#    )

# DEBUG, unhash as needed
print('LIR Estimates are:')
print(EstimatesLIR['DIC2'][300:310]) 
print('LIR Coefficient A estimates are:')
print(CoefficientsLIR['DIC2']['Coef A'][5:10])
print('NN Estimates are:')
print(EstimatesNN['phosphate3'][400:410])
print('NN uncertainties are:')
print(UncertaintiesNN['phosphate3'][501:503])
print('LIR uncertainties are:')
print(UncertaintiesLIR['TA3'][308:310])

# Optional format to pandas and save
#import pandas as pd
#(pd.DataFrame.from_dict(data=EstimatesLIR, orient='index')
#    .to_csv('TA1LIR.csv'))
#(pd.DataFrame.from_dict(data=EstimatesNN, orient='index')
#    .to_csv('EstNN.csv'))
