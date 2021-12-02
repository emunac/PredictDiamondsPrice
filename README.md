# PredictDiamondsPrice

Some insights from the results:

when predicting with two hidden layer 26->18->9->1
1. learning rate of 0.1 is too high - is often reaches very high validation error. Learning rate of 0.001 tend to be slightly better than 0.01. (TODO: check lower learning rate, e.g., 1e-4 or 3e-5). 
2. scale does not seem to be important. Model works well even without scaling (weird/worry - this is in contrast to conventional wisdom).

