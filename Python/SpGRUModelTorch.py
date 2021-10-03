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
def SpGRUModelTorch(data,Lag,sp,N,ep,spp = 0.1):
    import torch
    import torch.nn as nn
    import numpy as np
    from numpy import reshape
    from scipy.linalg import hankel
    
    md = data.min()
    
    Md = abs(data - md).max()
    
    data = (data-md)/Md
    
    L = int(len(data) * sp)
    
    xt = data[:L]
    
    H = hankel(xt[:Lag],xt[(Lag-1):]).T
    Xt = H[:-1,:]
    Yt = H[1:,-1]
    
    Xtt = []
    Ytt = []
    for k in range(Xt.shape[0]-1):
        Xtt.append(torch.FloatTensor(reshape(Xt[k],(1,1,Lag))))
        Ytt.append(torch.FloatTensor(reshape(Yt[k],(1,1))))
     
    class GRU(nn.Module):
        def __init__(self, num_classes, input_size, hidden_size, num_layers):
            super(GRU, self).__init__()
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.input_size = input_size
            self.hidden_size = hidden_size
           
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
            _, h_out = self.gru(x, h_0)
            h_out = h_out.view(-1, self.hidden_size)
            out = self.linear(h_out)   
            return out
    
    model = GRU(1,Lag,N,1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=spp)
    
    error = 1.0
    i = 0
    while i <= ep and error>10*spp:
        for k in range(len(Ytt)):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
            y_pred = model(Xtt[k])
            single_loss = loss_function(y_pred, Ytt[k])
            single_loss.backward()
            optimizer.step()
        error = single_loss.item()
    
    h = data[:Lag]
    
    return model,H,h