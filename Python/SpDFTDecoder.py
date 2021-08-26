#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:35:15 2021

@author: doctor
"""
def SpDFTDecoder(J,x):
    from numpy import arange,zeros,append
    from scipy.fft import ifft
    N = len(x)
    L = len(x[0])
    yr=[]
    z0 = zeros((L,))
    z0 = z0 + 1j*z0
    for j in arange(0,N):
        x0 = z0
        x0[J[j]]=x[j][J[j]]
        x0=ifft(x0)
        yr=append([yr],[x0])
    return yr
