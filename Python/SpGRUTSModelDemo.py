# Example: python3 SpGRUTSModelDemo.py '../DataSets/signal_with_anomaly.csv' 300 20 3400 0.15 16 0.3
# Example: runfile("SpGRUTSModelDemo.py","'../DataSets/signal_with_anomaly.csv' 10 20 3400 0.15 16 0.3")
# Example: runfile("SpGRUTSModelDemo.py","'../DataSets/signal_with_anomaly.csv' 10 60 880 0.2 16 0.3")
import sys
"""
Created on Wed Mar 31 02:57:52 2021
Sparse model demo based on sparse GRU modeling
   Code by Fredy Vides
   For Paper, "Computing Sparse Autoencoders and Autoregressors for Signal Identification"
   by F. Vides
@author: Fredy Vides
"""
import pandas
from matplotlib.pyplot import plot
from numpy import append,sin,pi,arange
from SpGRUModel import SpGRUModel
from SpGRUPredictor import SpGRUPredictor
from LagEstimate import LagEstimate


#data = pandas.read_csv(sys.argv[1], usecols=[1], engine='python')
#plot(data)

# data = data.values[:,0]

t = arange(0,10,.01)
t = append(t,10)
data = sin(2*pi*t)

Lag = LagEstimate(data,int(sys.argv[2]))
sp = float(sys.argv[5])
nn = int(sys.argv[6])
ep = int(sys.argv[3])
md = data.min()
Md = abs(data - md).max()
N = int(sys.argv[4])
spp = float(sys.argv[7])


TS_Model,H,h = SpGRUModel(data,Lag,sp,nn,ep,spp)
X = SpGRUPredictor(data,TS_Model,h,N)

plot(data[:len(X)]),plot(X)