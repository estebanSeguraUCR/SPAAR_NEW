#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:35:15 2021

@author: doctor
"""
def SpDFTEncoder(Y,S,L=100,tol=1e-2,delta=1e-2):
    import numpy as np
    from scipy.fft import fft
    from DFTSpSolver import DFTSpSolver
    y=[]
    J=[]
    N=int(np.floor(len(Y)/L))
    for j in np.arange(0,N):
        z0 = Y[j*L:((j+1)*L)]
        J0 = DFTSpSolver(z0,L,tol,delta)[0]
        y.append(fft(z0))
        J.append(J0)
    return J,y