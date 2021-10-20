#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
SPGRUMODEL GRU based sparse signal model
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
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
        Ytt.append(torch.FloatTensor(reshape(Yt[k],(1,1,1))))
    
    class GRU_Adapter(nn.Module):
        def __init__(self, index):
            super(GRU_Adapter, self).__init__()
            self._name = 'gru_adapter'
            self.index = index
        def forward(self, inputs):
            return inputs[self.index]
   
    model = torch.nn.Sequential(
        torch.nn.GRU(Lag,N,batch_first=True),
        GRU_Adapter(1),
        torch.nn.Linear(N,1)
        )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=spp)
    
    error = 1.0
    i = 0
    while i <= ep and error>10*spp:
        for k in range(len(Ytt)):
            optimizer.zero_grad()
            y_pred = model(Xtt[k])
            single_loss = loss_function(y_pred, Ytt[k])
            single_loss.backward()
            optimizer.step()
        error = single_loss.item()
    
    h = data[:Lag]
    
    return model,H,h