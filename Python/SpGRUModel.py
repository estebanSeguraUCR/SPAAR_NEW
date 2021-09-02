#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:57:52 2021
SPGRUMODEL GRU based sparse signal model
   Code by Fredy Vides
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
"""
def SpGRUModel(data,Lag,sp,nn,ep,spp = 0.1):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import GRU
    from sklearn.metrics import mean_squared_error
    from numpy import append,reshape,vstack,sin,pi,arange
    from scipy.linalg import hankel
    from keras.callbacks import EarlyStopping
    
    md = data.min()
    
    Md = abs(data - md).max()
    
    data = (data-md)/Md
    
    L = int(len(data) * sp)
    
    xt = data[:L]
    
    H = hankel(xt[:Lag],xt[(Lag-1):]).T
    
    
    Xt = H[0:(L-Lag-1),:]
    Yt = H[1:(L-Lag),Lag-1]
    
    Xt = reshape(Xt, (Xt.shape[0], 1, Xt.shape[1]))
    
    TS_Model = Sequential([
        GRU(nn, input_shape=(1, Lag),dropout=spp,recurrent_dropout=spp),
        Dense(1)
        ])
        
    TS_Model.compile(loss='mean_squared_error', optimizer='adam')
    
    callback = EarlyStopping(monitor='loss', mode='min', 
                         verbose=0, patience=10,min_delta=9e-4)
    
    TS_Model.fit(Xt, Yt, epochs=ep, batch_size=2, verbose=0,
             workers=6,use_multiprocessing=True,callbacks=[callback])
    
    h = data[:Lag]
    
    return TS_Model,H,h