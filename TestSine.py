#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:50:11 2022

@author: rmcclarr
"""
import numpy as np
import matplotlib.pyplot as plt

Nt = 100

dt = 0.1
Yminus = np.zeros((2,Nt-1))
Yplus = np.zeros((2,Nt-1))

dts = np.logspace(-3,-0.5,Nt)
#dts = 0.1*np.ones(Nt)
trange = np.cumsum(dts)
yval = np.zeros(2)
yval[0]=1
omega = 3.5**2
lam = -1e-1

#set up matrix
A = np.zeros((2,2))
A[0,1] = 1.0
A[1,0] = -omega
A[1,1] = lam
print("A's eigenvalues", np.linalg.eigvals(A))
#Backward Euler
for i in range(Nt-1):
    dt = dts[i]
    update = np.linalg.inv(np.identity(2) - dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval)
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:-1],Yminus[0,:])


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
    update = np.linalg.inv(np.identity(2) - 0.5*dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval + 0.5*dt*np.dot(A,yval))  
    Yminus[:,i] = 0.5*(yval + yold)
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:-1],Yminus[0,:])
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


#BDF-2

yval = np.zeros(2)
yval[0]=1
omega = 1.0
Yminus = np.zeros((2,Nt-1))
Yplus = np.zeros((2,Nt-1))

#annoying starting value; use C-N
i = 0
dt = dts[0]
update = np.linalg.inv(np.identity(2) - 0.5*dt*A)
yold = yval.copy()
yval = np.dot(update,yval + 0.5*dt*np.dot(A,yval))  
Yminus[:,i] = 0.5*(yval + yold)
Yplus[:,i] = (yval-yold)/dt

for i in range(1,Nt-1):
    dt = dts[i]
    yold2 = yold.copy()
    update = np.linalg.inv(np.identity(2) - 2/3*dt*A)
    yold = yval.copy()
    yval = np.dot(update,4/3*yval)  
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-4/3*yold+1/3*yold2)/(2/3*dt)
    
plt.plot(trange[0:-1],Yminus[0,:])
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