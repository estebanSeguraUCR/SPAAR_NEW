"""
Created on Wed Mar 31 02:57:52 2021
SPGRUPREDICTOR GRU based sparse predictor
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
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
    x0 = FloatTensor(reshape(h.copy(),(1,1,Lag)))
    for j in range(N):
        x = model(x0)
        xc = x0.clone()
        x0[0][0][:-1] = xc[0][0][1:]
        x0[0][0][-1]=x.detach()
        X = append(X,x.detach().numpy()[0][0])
    X = Md*X+md
    return X