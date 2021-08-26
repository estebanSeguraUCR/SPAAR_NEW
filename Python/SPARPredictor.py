#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 01:06:38 2021

@author: doctor
"""

def SPARPredictor(A,h,N):
    from numpy import arange,append
    y=h
    L = len(h)
    for k in arange(N):
        y = append(y,A.dot(h))
        h = y[k+1:L+k+1]
    return y