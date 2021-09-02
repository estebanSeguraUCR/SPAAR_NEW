#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:31:47 2021
Created on Wed Mar 31 02:57:52 2021
DFTSPSOLVER DFT based sparse linear regression solver
   Code by Fredy Vides
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
@author: doctor
"""
def LagEstimate(data,L0):
    from statsmodels.tsa.stattools import acf 
    from scipy.signal import find_peaks
    from numpy import where
    L = acf(data,adjusted=False,nlags=len(data),fft=True)
    p = find_peaks(L)
    q=where(L[p[0]]==L[p[0]].max())
    L = max(p[0][q[0][0]],L0)
    return L