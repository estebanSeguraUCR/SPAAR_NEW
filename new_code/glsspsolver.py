#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
LSSPSOLVER  Sparse linear least squares solver
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Nonlinear Systems"
   by F. Vides, Juan Pablo Soto and Esteban Segura
@authors: F. Vides, Juan Pablo Soto and Esteban Segura
"""

def lsspsolver(A,Y,L=100,tol=1e-2,delta=1e-2):
    from numpy.linalg import svd,lstsq,norm
    from numpy import zeros,dot,diag,argsort,sort,inf
    N=A.shape[1]
    X=zeros((N,))
    u,s,v=svd(A,full_matrices=0)
    rk=sum(s>tol)
    u=u[:,:rk]
    s=s[:rk]
    s=1/s
    s=diag(s)
    v=v[:rk,:]
    A=dot(u.T,A)
    Y=dot(u.T,Y)
    X0=dot(v.T,dot(s,Y))
    w=zeros((N,))
    K=1
    Error=1+tol
    c=X0
    x0=c
    ac=abs(c)
    f=argsort(-ac)
    N0=max(sum(ac[f]>delta),1)
    while (K<=L) & (Error>tol):
        ff=f[:N0]
        X=w
        c, res, rnk, s = lstsq(A[:,ff],Y,rcond=None)
        X[ff]=c
        Error=norm(x0-X[:],inf)
        x0=X
        ac=abs(x0)
        f=argsort(-ac)
        N0=max(sum(ac[f]>delta),1)
        K=K+1
    return X
