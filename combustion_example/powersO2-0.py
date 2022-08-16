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
y0 = np.array((0.001,0.001))

sol = solve_ivp(full_rhs, (0,1e-7),y0,max_step=1e-10, first_step=1e-14)
sol2 = solve_ivp(full_rhs, (0,1e-7),y0,max_step=1e-10, first_step=1e-14)
sol3 = solve_ivp(full_rhs, (0,1e-7),y0,max_step=1e-10, first_step=1e-14)
plt.semilogx(sol.t,sol.y[0,:], label = "O")
plt.semilogx(sol.t,sol.y[1,:], "--", label = "O$_2$")
plt.legend(loc="best")
print("O equil =",sol.y[0,-1],"ref is 0.0004424")
print("O2 equil =",sol.y[1,-1],"ref is 0.00127")
plt.show()

#try my linearized way

dts = sol.t[1:]-sol.t[:-1]
Nsteps = int(dts.size+1)
times = np.zeros(Nsteps)
sols = np.zeros((Nsteps,2))
eigs = np.zeros((Nsteps-1,2))
sols[0,:] = y0.copy()
Yminus = np.zeros((Nsteps,2))
Yplus = np.zeros((Nsteps,2))
lagDMD  = 2
lagDMD2 = 5
# lagDMD2 = 2
lagDMD3 = 10
# lagDMD3 = 2
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
plt.title('O-$O_2$ Concentration vs Time')
plt.savefig('OO2.pdf')
plt.show()

dts2 = sol2.t[1:]-sol2.t[:-1]
Nsteps2 = int(dts2.size+1)
times2 = np.zeros(Nsteps2)
sols2 = np.zeros((Nsteps2,2))
eigs2 = np.zeros((Nsteps2-1,2))
sols2[0,:] = y0.copy()
Yminus2 = np.zeros((Nsteps2,2))
Yplus2 = np.zeros((Nsteps2,2))
dmd_eigs2 = np.zeros((Nsteps2,2))
for step in range(1,Nsteps2):
    dt = dts2[step-1]
    idt = 1/dt
    J = jacobian(sols2[step-1])
    eigs2[step-1] = np.linalg.eigvals(J)
    sols2[step] = sols2[step-1] + np.linalg.solve(np.identity(2) - 0.5*dt*J,dt*full_rhs(0,sols2[step-1]))
    times2[step] = times2[step-1]+dt
    Yplus2[step] = idt*(sols2[step] - sols2[step-1])-full_rhs(0,sols2[step-1])
    Yminus2[step] = (sols2[step] - sols2[step-1])/2
    if (step>=lagDMD2):
        dmd_eigs2[step] = simpleDMD(Yminus2[(step-lagDMD2):step,:],Yplus2[(step-lagDMD2):step,:])

dts3 = sol3.t[1:]-sol3.t[:-1]
Nsteps3 = int(dts3.size+1)
times3 = np.zeros(Nsteps3)
sols3 = np.zeros((Nsteps3,2))
eigs3 = np.zeros((Nsteps3-1,2))
sols3[0,:] = y0.copy()
Yminus3 = np.zeros((Nsteps3,2))
Yplus3 = np.zeros((Nsteps3,2))

dmd_eigs3 = np.zeros((Nsteps3,2))
for step in range(1,Nsteps3):
    dt = dts3[step-1]
    idt = 1/dt
    J = jacobian(sols3[step-1])
    eigs3[step-1] = np.linalg.eigvals(J)
    sols3[step] = sols3[step-1] + np.linalg.solve(np.identity(2) - 0.5*dt*J,dt*full_rhs(0,sols3[step-1]))
    times3[step] = times3[step-1]+dt
    Yplus3[step] = idt*(sols3[step] - sols3[step-1])-full_rhs(0,sols3[step-1])
    Yminus3[step] = (sols3[step] - sols3[step-1])/2
    if (step>=lagDMD3):
        dmd_eigs3[step] = simpleDMD(Yminus3[(step-lagDMD3):step,:],Yplus3[(step-lagDMD3):step,:])

print("O equil =",sols[-1,0],"ref is 0.0004424")
print("O2 equil =",sols[-1,1],"ref is 0.00127")
plt.show()
plt.loglog(times[:-1],np.abs(eigs[:,0]),label='Analytic')
# plt.loglog(times,np.abs(eigs[:,1]))
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,1],".",color='r')
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,0],".",label='VDMD, lag = {}'.format(lagDMD),color='r')
plt.loglog(times2[lagDMD2:],np.abs(dmd_eigs2)[lagDMD2:,1],".",color='g')
plt.loglog(times2[lagDMD2:],np.abs(dmd_eigs2)[lagDMD2:,0],".",label='VDMD, lag = {}'.format(lagDMD2),color='g')
plt.loglog(times3[lagDMD3:],np.abs(dmd_eigs3)[lagDMD3:,1],".",color='orange')
plt.loglog(times3[lagDMD3:],np.abs(dmd_eigs3)[lagDMD3:,0],".",label='VDMD, lag = {}'.format(lagDMD3),color='orange')
plt.xlabel('Time, (s)')
plt.ylabel(r'Eigenvalue, (s$^{-1}$)')
plt.title('VDMD Eigenvalue vs Analytic Eigenvalue')
plt.legend()
plt.savefig('oo2eig_varlag.pdf')
plt.show()
#plt.loglog(times,np.abs(eigs[:,1]))

y0 = np.array((0.01,1e-4))
y0 = np.array((0.001,0.001))

max_0 = 1e-11
max_1 = 1e-10
max_2 = 1e-9

sol = solve_ivp(full_rhs, (0,1e-7),y0,max_step=max_0, first_step=1e-14)
sol2 = solve_ivp(full_rhs, (0,1e-7),y0,max_step=max_1, first_step=1e-14)
sol3 = solve_ivp(full_rhs, (0,1e-7),y0,max_step=max_2, first_step=1e-14)
plt.semilogx(sol.t,sol.y[0,:], label = "O")
plt.semilogx(sol.t,sol.y[1,:], "--", label = "O$_2$")
plt.legend(loc="best")
print("O equil =",sol.y[0,-1],"ref is 0.0004424")
print("O2 equil =",sol.y[1,-1],"ref is 0.00127")
plt.show()

#try my linearized way

dts = sol.t[1:]-sol.t[:-1]
Nsteps = int(dts.size+1)
times = np.zeros(Nsteps)
sols = np.zeros((Nsteps,2))
eigs = np.zeros((Nsteps-1,2))
sols[0,:] = y0.copy()
Yminus = np.zeros((Nsteps,2))
Yplus = np.zeros((Nsteps,2))
lagDMD  = 2
lagDMD2 = 2
lagDMD3 = 2
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
plt.title('O-$O_2$ Concentration vs Time')
plt.savefig('OO2.pdf')
plt.show()

dts2 = sol2.t[1:]-sol2.t[:-1]
Nsteps2 = int(dts2.size+1)
times2 = np.zeros(Nsteps2)
sols2 = np.zeros((Nsteps2,2))
eigs2 = np.zeros((Nsteps2-1,2))
sols2[0,:] = y0.copy()
Yminus2 = np.zeros((Nsteps2,2))
Yplus2 = np.zeros((Nsteps2,2))
dmd_eigs2 = np.zeros((Nsteps2,2))
for step in range(1,Nsteps2):
    dt = dts2[step-1]
    idt = 1/dt
    J = jacobian(sols2[step-1])
    eigs2[step-1] = np.linalg.eigvals(J)
    sols2[step] = sols2[step-1] + np.linalg.solve(np.identity(2) - 0.5*dt*J,dt*full_rhs(0,sols2[step-1]))
    times2[step] = times2[step-1]+dt
    Yplus2[step] = idt*(sols2[step] - sols2[step-1])-full_rhs(0,sols2[step-1])
    Yminus2[step] = (sols2[step] - sols2[step-1])/2
    if (step>=lagDMD2):
        dmd_eigs2[step] = simpleDMD(Yminus2[(step-lagDMD2):step,:],Yplus2[(step-lagDMD2):step,:])

dts3 = sol3.t[1:]-sol3.t[:-1]
Nsteps3 = int(dts3.size+1)
times3 = np.zeros(Nsteps3)
sols3 = np.zeros((Nsteps3,2))
eigs3 = np.zeros((Nsteps3-1,2))
sols3[0,:] = y0.copy()
Yminus3 = np.zeros((Nsteps3,2))
Yplus3 = np.zeros((Nsteps3,2))

dmd_eigs3 = np.zeros((Nsteps3,2))
for step in range(1,Nsteps3):
    dt = dts3[step-1]
    idt = 1/dt
    J = jacobian(sols3[step-1])
    eigs3[step-1] = np.linalg.eigvals(J)
    sols3[step] = sols3[step-1] + np.linalg.solve(np.identity(2) - 0.5*dt*J,dt*full_rhs(0,sols3[step-1]))
    times3[step] = times3[step-1]+dt
    Yplus3[step] = idt*(sols3[step] - sols3[step-1])-full_rhs(0,sols3[step-1])
    Yminus3[step] = (sols3[step] - sols3[step-1])/2
    if (step>=lagDMD3):
        dmd_eigs3[step] = simpleDMD(Yminus3[(step-lagDMD3):step,:],Yplus3[(step-lagDMD3):step,:])

print("O equil =",sols[-1,0],"ref is 0.0004424")
print("O2 equil =",sols[-1,1],"ref is 0.00127")
plt.show()
plt.loglog(times[:-1],np.abs(eigs[:,0]),label='Analytic')
# plt.loglog(times,np.abs(eigs[:,1]))
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,1],".",color='r')
plt.loglog(times[lagDMD:],np.abs(dmd_eigs)[lagDMD:,0],".",label='VDMD, max dt = {}'.format(max_0),color='r')
plt.loglog(times2[lagDMD2:],np.abs(dmd_eigs2)[lagDMD2:,1],".",color='g')
plt.loglog(times2[lagDMD2:],np.abs(dmd_eigs2)[lagDMD2:,0],".",label='VDMD, max dt = {}'.format(max_1),color='g')
plt.loglog(times3[lagDMD3:],np.abs(dmd_eigs3)[lagDMD3:,1],".",color='orange')
plt.loglog(times3[lagDMD3:],np.abs(dmd_eigs3)[lagDMD3:,0],".",label='VDMD, max dt = {}'.format(max_2),color='orange')
plt.xlabel('Time, (s)')
plt.ylabel(r'Eigenvalue, (s$^{-1}$)')
plt.title('VDMD Eigenvalue vs Analytic Eigenvalue')
plt.legend()
plt.savefig('oo2eig_vardt.pdf')
plt.show()