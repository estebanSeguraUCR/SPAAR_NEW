#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:57:52 2021
DFTSPSOLVER DFT based sparse linear regression solver
   Code by Fredy Vides
   For Paper, "On Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
"""

def DFTSpSolver(Y,L=100,tol=1e-2,delta=1e-2):
    from numpy.linalg import svd,lstsq,norm
    from numpy import zeros,dot,diag,argsort,sort,inf
    from scipy.fft import fft,ifft
    N=Y.shape[0]
    X=zeros((N,))
    X=X+1j*X
    w=X
    K=1
    Error=1+tol
    c=fft(Y)
    x0=c
    ac=abs(c)
    f=argsort(-ac)
    N0=max(sum(ac[f]>delta),1)
    while (K<=L) & (Error>tol):
        J=f[:N0]
        X=w
        X[J]=c[J]
        Error=norm(ifft(x0-X),inf)
        x0=X
        ac=abs(x0)
        f=argsort(-ac)
        N0=max(sum(ac[f]>delta),1)
        K=K+1
    return J,X