#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:42:00 2022

@author: rmcclarr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:50:11 2022

@author: rmcclarr
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

Nt = 2000

dt = 0.1
Yminus = np.zeros((2,Nt-1))
Yplus = np.zeros((2,Nt-1))

dts = np.logspace(-4,-1,Nt)
#dts = 0.1*np.ones(Nt)
trange = np.cumsum(dts)
yval = np.zeros(2)
yval[0]=1
omega = 1
lam = -1e-1

#set up matrix
A = np.zeros((2,2))
A[0,1] = 1.0
A[1,0] = -omega
A[1,1] = 0
print("A's eigenvalues", np.linalg.eigvals(A))
#Backward Euler
for i in range(Nt-1):
    dt = dts[i]
    A[1,1] = -1/trange[i+1]
    update = np.linalg.inv(np.identity(2) - dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval)
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:-1],Yminus[0,:])
plt.plot(trange,jv(0,trange))


skip = 0

[U,S,V] = svd(Yminus[:,skip:],full_matrices=False)
Sinv = np.zeros(S.size)
Spos = S[S/np.cumsum(S)>1e-18]
Sinv[0:Spos.size] = 1.0/Spos.copy()
tmp=np.dot(U.transpose(),Yplus[:, skip:])
tmp2=np.dot(tmp,V.transpose())
tmp3=np.dot(tmp2,np.diag(Sinv))
deigs = np.linalg.eigvals(tmp3)
#deigs = deigs[deigs>0]
#print(np.log(deigs)/dt)
print(deigs)

#Crank-Nicolson

yval = np.zeros(2)
yval[0]=1
omega = 1.0
Yminus = np.zeros((2,Nt-1))
Yplus = np.zeros((2,Nt-1))

for i in range(Nt-1):
    dt = dts[i]
    A[1,1] = -2/(trange[i+1] + trange[i])
    update = np.linalg.inv(np.identity(2) - 0.5*dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval + 0.5*dt*np.dot(A,yval))  
    Yminus[:,i] = 0.5*(yval + yold)
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:-1],Yminus[0,:],'--')
[U,S,V] = svd(Yminus[:,skip:],full_matrices=False)
Sinv = np.zeros(S.size)
Spos = S[S/np.cumsum(S)>1e-18]
Sinv[0:Spos.size] = 1.0/Spos.copy()
tmp=np.dot(U.transpose(),Yplus[:, skip:])
tmp2=np.dot(tmp,V.transpose())
tmp3=np.dot(tmp2,np.diag(Sinv))
deigs = np.linalg.eigvals(tmp3)
#deigs = deigs[deigs>0]
#print(np.log(deigs)/dt)
print(deigs)

