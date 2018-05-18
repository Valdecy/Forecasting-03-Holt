############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Forecasting
# Lesson: Holt

# Citation: 
# PEREIRA, V. (2018). Project: Forecasting, File: Python-Forecasting-03-Holt.py, GitHub repository: <https://github.com/Valdecy/Forecasting-03-Holt>

############################################################################

# Installing Required Libraries
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import mean_squared_error
from math import sqrt

################     Part 1 - Holt's Method    #############################

# Function: Holt
def holt(timeseries, alpha = 0.2, beta = 0.1, graph = True, horizon = 0, trend = "multiplicative"):
   
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    holt   = pd.DataFrame(np.nan, index = timeseries.index, columns = ['Holt'])
    holt_A = pd.DataFrame(np.nan, index = timeseries.index, columns = ['A'])
    holt_T = pd.DataFrame(np.nan, index = timeseries.index, columns = ['T'])
    n = 1

    for i in range(0, len(timeseries) - n):
        
        if (i == 0 and trend == "none"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0]
          
        elif (i == 0 and trend == "additive"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt_T.iloc[i, 0] = 0.0 
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
          
        elif (i == 0 and trend == "multiplicative"):
            holt_A.iloc[i, 0] = float(timeseries.iloc[0,:])
            holt_T.iloc[i, 0] = 1.0
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
          
        elif (i > 0 and trend == "none"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0])
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0]
            last = float(holt.iloc[i,0])

        elif (i > 0 and trend == "additive"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0] + holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] - holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] + n*holt_T.iloc[i, 0]
            last = float(holt.iloc[i,0])
            
        elif (i > 0 and trend == "multiplicative"):
            holt_A.iloc[i, 0]  = alpha*(float(timeseries.iloc[i,:])) + (1 - alpha)*(holt_A.iloc[i - 1, 0] * holt_T.iloc[i - 1, 0])
            holt_T.iloc[i, 0]  = beta*(holt_A.iloc[i, 0] / holt_A.iloc[i - 1, 0]) + (1 - beta)*holt_T.iloc[i - 1, 0]
            holt.iloc[i + n, 0]  =  holt_A.iloc[i, 0] * n*holt_T.iloc[i, 0]
            last = float(holt.iloc[i,0])
    
    if horizon > 0: 
        time_horizon = len(timeseries) + horizon 
        time_horizon_index = pd.date_range(timeseries.index[0], periods = time_horizon, freq = timeseries.index.inferred_freq) 
        pred = pd.DataFrame(np.nan, index = time_horizon_index, columns = ["Prediction"])
        for i in range(0, horizon):
            pred.iloc[len(timeseries) + i] = last
        pred = pred.iloc[:,0]
    
    rms = sqrt(mean_squared_error(timeseries.iloc[(n+1):,0], holt.iloc[(n+1):,0]))
    timeseries = timeseries.iloc[:,0]
    holt = holt.iloc[:,0]
    
    if graph == True and horizon <= 0:
        style.use('ggplot')
        plt.plot(timeseries)
        plt.plot(holt)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.xticks(rotation = 90)
        plt.show()
    elif graph == True and horizon > 0:
        style.use('ggplot')
        plt.plot(timeseries)
        plt.plot(holt)
        plt.plot(pred)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.xticks(rotation = 90)
        plt.show()
   
    return holt, last, rms

    ############### End of Function ##############

# Brute Force Optmization    
def optimize_holt(timeseries, trend = "multiplicative"):
    error = pd.DataFrame(columns = ['alpha', 'beta', 'rmse'])
    count = 0
    for alpha in range(0, 101):
        for beta in range(0, 101):
            print("alpha = ", alpha/100, " beta = ", beta/100)
            ts, last, rms = holt(timeseries, alpha = alpha/100, beta = beta/100, graph = False, horizon = 0, trend = trend)
            error.loc[count] = [alpha/100, beta/100, rms]
            count = count + 1
    return error, error.loc[error['rmse'].idxmin()]

    ############### End of Function ##############

######################## Part 2 - Usage ####################################
 
# Load Dataset 
df = pd.read_csv('Python-Forecasting-03-Dataset.txt', sep = '\t')

# Transform Dataset to a Time Series
X = df.iloc[:,:]
X = X.set_index(pd.DatetimeIndex(df.iloc[:,0])) # First column as row names
X = X.iloc[:,1]

# Calling Functions
holt(X, alpha = 0.61, beta = 0.55, graph = True, horizon = 0, trend = "multiplicative")

optimize_holt(X, trend = "multiplicative")

########################## End of Code #####################################
