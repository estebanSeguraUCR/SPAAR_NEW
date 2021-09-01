#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:57:52 2021
SPENCODER based on sparse dense layers
   Code by Fredy Vides
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
"""
def SpDenseAutoencoder(data,k,L,N,ep,sp):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras import regularizers
    from scipy.linalg import hankel
    from keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_squared_error
    from numpy import ceil
    
    h = hankel(data[:L],data[(L-1):]).T
    h0 = h[:(L-1),:]
    h1 = h[1:L,:]
    
    tdata = data[:N]
    
    model = Sequential([
            Dropout(sp[0],input_shape=(L,)),
            Dense(k,activation='linear',
                  activity_regularizer=regularizers.l1(1e-5)),
            Dropout(sp[1]),
            Dense(L, activation='linear')
        ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    callback = EarlyStopping(monitor='loss', mode='min', 
                         verbose=0, patience=10,min_delta=9e-4)
    
    from keras.callbacks import EarlyStopping
    
    model.fit(h,h,epochs=ep, batch_size=2, verbose=0,
             workers=6,use_multiprocessing=True,callbacks=[callback])
    
    h = data[:L]
    
    return model,h