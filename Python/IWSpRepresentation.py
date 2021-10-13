"""
Created on Wed Mar 31 02:57:52 2021
INPUTSPREPRESENTATION Sparse representation of input weights of the GRU model
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def IWSpRepresentation(H,model,delta,tol):
    from lsspsolver import lsspsolver
    from numpy import array
    W = model.layers[0].get_weights()
    X01 = H@W[0]
    U = []
    for k in range(W[0].shape[1]):
        w1 = lsspsolver(H,X01[:,k],W[0].shape[0],delta,5e-3)
        U.append(w1)
    U=array(U)
    W[0] = U.T
    model.layers[0].set_weights(W)
    return model