#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:55:44 2022

@author: rmcclarr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


a11 = -1.16e14
a12 = 1.77548e10
a21 = 5.8e13
a22 = - 8.8774e9
full_rhs = lambda t,rhoV: np.array([(a11 * rhoV[0]**2 * (rhoV[0] + rhoV[1]) + a12*rhoV[1]*(rhoV[0] + rhoV[1])),
                           (a21 * rhoV[0]**2 * (rhoV[0] + rhoV[1]) + a22*rhoV[1]*(rhoV[0] + rhoV[1]))])

def simpleDMD(Yminus,Yplus):
    [U,S,V] = np.linalg.svd(Yminus,full_matrices=False)
    Sinv = np.zeros(S.size)
    if (np.max(np.abs(S))>1e-12):
        Spos = S[S/np.cumsum(S)>1e-6]
        Sinv[0:Spos.size] = 1.0/Spos.copy()
        tmp=np.dot(U.transpose(),Yplus)
        tmp2=np.dot(tmp,V.transpose())
        tmp3=np.dot(tmp2,np.diag(Sinv))
        return np.linalg.eigvals(tmp3)
    else:
        return np.zeros(2)

def jacobian(rhoV):
    o = rhoV[0]
    o2 = rhoV[1]
    J = np.zeros((2,2))
    J[0,0] = a11*o**2 + a12*o2 + 2*a11*o*(o + o2)
    J[0,1] = a11*o**2 + a12*o2 + a12*(o + o2)
    J[1,0] = a21*o**2 + a22*o2 + 2*a21*o*(o + o2)
    J[1,1] = a21*o**2 + a22*o2 + a22*(o + o2)
    return J #.transpose()

def fdJacobian(rhoV):
    o = rhoV[0]
    o2 = rhoV[1]
    J = np.zeros((2,2))
    delta = 1e-6
    J[0,0], J[1,0] = np.imag((full_rhs(0,rhoV+np.array((1.0j*delta,0.0))))/delta)
    J[0,1], J[1,1] = np.imag((full_rhs(0,rhoV+np.array((0.0,1.0j*delta))))/delta)
    return J
    
y0 = np.array((0.01,1e-4))

sol = solve_ivp(full_rhs, (0,1e-6),y0,max_step=1e-10, first_step=1e-14)
plt.semilogx(sol.t,sol.y[0,:], label = "O")
plt.semilogx(sol.t,sol.y[1,:], "--", label = "O$_2$")
plt.legend(loc="best")
print("O equil =",sol.y[0,-1],"ref is 0.0004424")
print("O2 equil =",sol.y[1,-1],"ref is 0.00127")
#plt.show()

#try my linearized way

dts = sol.t[1:]-sol.t[:-1]
Nsteps = int(dts.size+1)
times = np.zeros(Nsteps)
sols = np.zeros((Nsteps,2))
eigs = np.zeros((Nsteps,2))
sols[0,:] = y0.copy()
Yminus = np.zeros((Nsteps,2))
Yplus = np.zeros((Nsteps,2))
lagDMD = 2
dmd_eigs = np.zeros((Nsteps,2))
for step in range(1,Nsteps):
    dt = dts[step-1]
    idt = 1/dt
    J = jacobian(sols[step-1])
    eigs[step-1] = np.linalg.eigvals(J)
    sols[step] = sols[step-1] + np.linalg.solve(np.identity(2) - 0.5*dt*J,dt*full_rhs(0,sols[step-1]))
    times[step] = times[step-1]+dt
    Yplus[step] = idt*(sols[step] - sols[step-1])-full_rhs(0,sols[step-1])
    Yminus[step] = (sols[step] - sols[step-1])/2
    if (step>=lagDMD):
        dmd_eigs[step] = simpleDMD(Yminus[(step-lagDMD):step,:],Yplus[(step-lagDMD):step,:])
plt.semilogx(times,sols[:,0], label = "O")
plt.semilogx(times,sols[:,1], "--", label = "O$_2$")
plt.legend(loc="best")
    
print("O equil =",sols[-1,0],"ref is 0.0004424")
print("O2 equil =",sols[-1,1],"ref is 0.00127")
plt.show()
plt.loglog(times,np.abs(eigs[:,0]))
#plt.loglog(times,np.abs(eigs[:,1]))
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,1],".")
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,0],".")
#plt.loglog(times,np.abs(eigs[:,1]))