#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:57:52 2021
SPAUTOENCODER  Sparse linear least squares autoencoder
   Code by Fredy Vides
   For Paper, "Computing sparse autoencoders and autoregressors for signal
   identification"
   by F. Vides
@author: Fredy Vides

Example:
from pandas import read_csv
from matplotlib.pyplot import plot, stem, show
from numpy import reshape
data = pandas.read_csv('../DataSets/signal_with_anomaly.csv', usecols=[1], 
                       engine='python')
data = data.values[:,0]
from SpAutoencoder import SpAutoencoder
W = SpAutoencoder(data,300,600,1e2)
h = reshape(data[:300],(300,1))
t = reshape(data[2900:(2900+300)],(300,1))
hw = W.T@h
tw  = W.T@t
plot(W@hw),plot(h)
show()
plot(W@hw),plot(h)
show()
stem(abs(hw))
show()
stem(abs(tw))
"""
def SpAutoencoder(data,L,N,tol=1e-2):
    from numpy.linalg import svd,lstsq,norm
    from numpy import zeros,dot,diag,argsort,sort,inf
    from scipy.linalg import hankel
    data = data[:N]
    H = hankel(data[:L],data[(L-1):])
    u,s,v=svd(H,full_matrices=0)
    rk=sum(s>tol)
    u=u[:,:rk]
    return u