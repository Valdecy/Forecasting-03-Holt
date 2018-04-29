# Forecasting-03-Holt

This Function is an Implementation of the Holt Method for Time Series with Trend. If Necessary it Can Also Return the Best values for Alpha and Beta.

* timeseries = The dataset in a Time Series format.

* alpha = Level smoothing parameter. The default value is 0.2

* beta = Trend smoothing parameter. The default value is 0.1

* graph = If True then the original dataset and the moving average curves will be plotted. The default value is True.

* horizon = Calculates the prediction h steps ahead. The default value is 0.

* trend = Indicates the types of trend: "additive", "multiplicative" or "none". The default value is "multiplicative"

* optimize = If True then the best "alpha" and "beta" are calculated by brute force. The default value is False.
