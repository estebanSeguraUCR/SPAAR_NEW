# Example: python3 LSTMTSModel.py '../DataSets/signal_with_anomaly.csv' 300 20 3400 0.15 16
# Example: runfile("LSTMTSModel.py","'../DataSets/signal_with_anomaly.csv' 300 20 3400 0.15 16")
# Example: runfile("LSTMTSModel.py","'../DataSets/signal_with_anomaly.csv' 30 60 880 0.1 16")
import sys
"""
Created on Wed Mar 31 02:57:52 2021
SPGRUPREDICTOR GRU based sparse predictor
   Code by Fredy Vides
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
"""
def SpGRUPredictor(data,model,h,N):
    from numpy import append,reshape
    Lag = len(h)
    md = data.min()
    Md = abs(data - md).max()
    X = []
    x0 = []
    X = h
    x0 = reshape(h,(1,1,Lag))
    for j in range(N):
        x = model.predict(x0)
        x0[:1,:,:][0][0][0:(Lag-1)]=x0[:1,:,:][0][0][1:]
        x0[:1,:,:][0][0][Lag-1]=x
        X = append(X,x)
    X = append(data[0],Md*X+md)
    return X