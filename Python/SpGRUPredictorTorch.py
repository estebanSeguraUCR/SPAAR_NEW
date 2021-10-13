<<<<<<< HEAD
=======
# Example: python3 LSTMTSModel.py '../DataSets/signal_with_anomaly.csv' 300 20 3400 0.15 16
# Example: runfile("LSTMTSModel.py","'../DataSets/signal_with_anomaly.csv' 300 20 3400 0.15 16")
# Example: runfile("LSTMTSModel.py","'../DataSets/signal_with_anomaly.csv' 30 60 880 0.1 16")
import sys
>>>>>>> 0bc613ea8fe7f2056b63b55a534552d93958825b
"""
Created on Wed Mar 31 02:57:52 2021
SPGRUPREDICTOR GRU based sparse predictor
   Code by Fredy Vides
<<<<<<< HEAD
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
=======
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
>>>>>>> 0bc613ea8fe7f2056b63b55a534552d93958825b
   by F. Vides
@author: Fredy Vides
"""
def SpGRUPredictorTorch(data,model,h,N):
    from numpy import append,reshape
    from torch import FloatTensor
    Lag = len(h)
    md = data.min()
    Md = abs(data - md).max()
    X = []
    x0 = []
    X = h
<<<<<<< HEAD
    x0 = FloatTensor(reshape(h.copy(),(1,1,Lag)))
=======
    x0 = FloatTensor(reshape(h,(1,1,Lag)))
>>>>>>> 0bc613ea8fe7f2056b63b55a534552d93958825b
    for j in range(N):
        x = model(x0)
        xc = x0.clone()
        x0[0][0][:-1] = xc[0][0][1:]
        x0[0][0][-1]=x.detach()
        X = append(X,x.detach().numpy()[0][0])
<<<<<<< HEAD
    X = Md*X+md
=======
    X = append(data[0],Md*X+md)
>>>>>>> 0bc613ea8fe7f2056b63b55a534552d93958825b
    return X