#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:14:29 2021

@author: rmcclarr
"""
import matplotlib.pyplot as plt
from multigroup_sn import *
from alphaCalc import *
import math
import numpy as np
from plotting import *

#set up the problem and run it
G = 1
L = 1.0
cells = 20
N = 16
I = int(np.round(cells*L)) #540
hx = L/I
q = np.ones((I,G))*0
Xs = np.linspace(hx/2,L-hx/2,I)
sigma_t = np.ones((I,G))*10
nusigma_f = np.zeros((I,G))
chi = np.ones((I,G))
sigma_s = np.ones((I,G,G))*10
sigma_s[Xs>L/2] = 0.9*10
inv_speed = 1


MU, W = np.polynomial.legendre.leggauss(N)
BCs = np.zeros((N,G)) 
psi0 = np.random.random((I,N,G)) + 1e-12
numsteps = 5
ts = np.logspace(-1,math.log10(500),numsteps + 1)
dt = np.diff(ts)
print("times =",ts)
print("delta t =",dt)
x,phi,psi = multigroup_td_var(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,dt, tolerance = 1.0e-8,maxits = 400000, LOUD=1 )
plt.plot(x,phi[:,0,-1])
plt.show()

np.savez("asymmetric-super_%i_%i.npz" %(cells,N), psi)

#get eigenvalues
step = 0
included = numsteps+1
eigsN, vsN,u = compute_alpha(psi[:,:,:,step:(step+included+1)],0,included,I,G,N,dt)
print(vsN.shape,u.shape)
print(eigsN[ np.abs(np.imag(eigsN)) < 1e-6])

np.savetxt("asymmetric-super_%i_%i.csv" %(cells,N), eigsN[ np.abs(np.imag(eigsN)) < 1e-6], delimiter=",")
#plot eigenvectors
evect = np.reshape(np.dot(u[:,0:vsN.shape[0]],vsN[:,np.argsort(eigsN.real)[-1]]),(I,N,G))
phi_mat = evect[:,0]*0
for angle in range(N):
    phi_mat +=  evect[:,angle]*W[angle]
    
evect = np.reshape(np.dot(u[:,0:vsN.shape[0]],vsN[:,np.argsort(eigsN.real)[-2]]),(I,N,G))
phi_mat2 = evect[:,0]*0
for angle in range(N):
    phi_mat2 +=  evect[:,angle]*W[angle]
    
    

signval = np.sign(phi_mat[np.argmax(np.abs(phi_mat))].real)[0,0]
plt.plot(x,signval*np.real(phi_mat)/np.max(np.abs(phi_mat)),label="Fundamental Mode")
#plt.plot(fund_new[:,0],fund_new[:,1]/np.max(np.abs(fund[:,1])),"--")
signval = np.sign(phi_mat2[np.argmax(np.abs(phi_mat2))])[0,0]
plt.plot(x,signval*np.real(phi_mat2)/np.max(np.abs(phi_mat2)),"--",label="Second Mode")
#plt.plot(sec_new[:,0],sec_new[:,1]/np.max(np.abs(sec_new[:,1])),"-.")
plt.legend(loc="best")
plt.xlabel("x (cm)")
plt.ylabel("Normalized eigenvector")
show("asymmetric-super_%i_%i.jpg" %(cells,N))