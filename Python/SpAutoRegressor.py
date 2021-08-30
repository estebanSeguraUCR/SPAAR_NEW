#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:57:52 2021
SPAUTOREGRESSOR  Sparse autoregressor for time series modeling
   Code by Fredy Vides
   For Paper, "Computing sparse autoencoders and autoregressors for signal
   identification"
   by F. Vides
@author: Fredy Vides
"""
# Example:
# from pandas import read_csv
# from SpAutoRegressor import SpAutoRegressor
# from SPARPredictor import SPARPredictor
# from numpy import ceil
# import matplotlib.pyplot as plt
# data = read_csv('../DataSets/signal_with_anomaly.csv', usecols=[1], engine='python')
# x = data.values
# mx = x.min()
# Mx = abs(x-mx).max()
# xs = (x-mx)/Mx
# A,h = SpAutoRegressor(xs,1/len(xs),.5,1,600,1e-1,1e-1)
# y = Mx*SPARPredictor(A,h,1800)+mx
# L0 = int(ceil(.5*len(xs)))
# plt.plot(x[(L0-600):L0+1800,0]),plt.plot(y)
def SpAutoRegressor(x,ssp,sp,pp,L0,tol,delta):
    from numpy import ceil,floor,max,min,asmatrix
    from scipy.linalg import hankel
    from lsspsolver import lsspsolver
    sl = len(x)
    ssp=int(ceil(sl*ssp))
    x = x[0:sl:ssp]
    sl = len(x)
    sp = int(ceil(sp*sl))
    xt = x[:sp]
    pp = max([min([floor(pp*sp),sl-sp]),L0])
    xl = x[(sp+1):]
    L=L0
    H=hankel(xt[:L],xt[(L-1):])
    Lh=H.shape[1]
    H0=H[:,:(Lh-1)]
    H1=H[L-1,1:Lh]
    A = lsspsolver(H0.T,H1.T,L,tol,delta)
<<<<<<< HEAD
    return A.T,H1[Lh-L-1:Lh]
=======
    return A.T,H1[Lh-L-1:Lh]
>>>>>>> 506aabc9d33f65b6a1302d9fb568f54028085d30
