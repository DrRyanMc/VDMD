#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 19:48:32 2022

@author: rmcclarr
"""

"""
Created on Mon Jun  6 13:55:44 2022

@author: rmcclarr
"""
from   numba import njit, jit, float64,int32
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import cg
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
@njit
def diffusion_update(t,u,Nx,Ny,dx,dy,G,D,vx,vy,geom):
    """
    Compute the RHS of the diffusion equation
    D is length G
    """
    idx = 1.0/dx
    idy = 1.0/dy
    idx2 = idx/dx
    idy2 = idy/dy
    uold = u.reshape((Nx,Ny,G))
    unew = np.zeros(uold.shape)
    #bcs are symmetry
    for i in range(Nx):
        for j in range(Ny):
            if (i>0):
                fleft = uold[i,j,:] - uold[i-1,j,:]
            else:
                fleft = np.zeros(G)
            if (i<Nx-1):
                fright = uold[i+1,j,:] - uold[i,j,:]
            else:
                fright = np.zeros(G)
            if (j>0):
                fbottom = uold[i,j,:] - uold[i,j-1,:]
            else:
                fbottom = np.zeros(G)
            if (j<Ny-1):
                ftop = uold[i,j+1,:] - uold[i,j,:]
            else:
                ftop = np.zeros(G)
            for g in range(G):
                unew[i,j,g] += D[g]*idx2*(fright[g] - fleft[g]) - vx[g]*idx*((vx[g]>0)*fleft[g] + (vx[g]<= 0)*fright[g])
                unew[i,j,g] += D[g]*idy2*(ftop[g] - fbottom[g]) - vy[g]*idy*((vy[g]>0)*fbottom[g] + (vy[g]<= 0)*ftop[g])
    return unew.reshape(unew.size)



@jit(float64[:](float64,float64[:]))
def full_reaction(t,rhoV):
    a11 = -1.16e14
    a12 = 1.77548e10
    a21 = 5.8e13
    a22 = - 8.8774e9
    return np.array([(a11 * rhoV[0]**2 * (rhoV[0] + rhoV[1]) + a12*rhoV[1]*(rhoV[0] + rhoV[1])),
                     (a21 * rhoV[0]**2 * (rhoV[0] + rhoV[1]) + a22*rhoV[1]*(rhoV[0] + rhoV[1]))])


def jacobian(rhoV):
    a11 = -1.16e14
    a12 = 1.77548e10
    a21 = 5.8e13
    a22 = - 8.8774e9
    o = rhoV[0]
    o2 = rhoV[1]
    J = np.zeros((2,2))
    J[0,0] = a11*o**2 + a12*o2 + 2*a11*o*(o + o2)
    J[0,1] = a11*o**2 + a12*o2 + a12*(o + o2)
    J[1,0] = a21*o**2 + a22*o2 + 2*a21*o*(o + o2)
    J[1,1] = a21*o**2 + a22*o2 + a22*(o + o2)
    return J

@njit
def applyJacobian(u,app):
    output = np.zeros(u.size).reshape((Nx,Ny,G))
    tmp = u.reshape((Nx,Ny,G))
    tmpapp = app.reshape((Nx,Ny,G))
    a11 = -1.16e14
    a12 = 1.77548e10
    a21 = 5.8e13
    a22 = - 8.8774e9
    J = np.zeros((2,2))
    for i in range(Nx):
        for j in range(Ny):
            o = tmp[i,j,0]
            o2 = tmp[i,j,1]
            J[0,0] = a11*o**2 + a12*o2 + 2*a11*o*(o + o2)
            J[0,1] = a11*o**2 + a12*o2 + a12*(o + o2)
            J[1,0] = a21*o**2 + a22*o2 + 2*a21*o*(o + o2)
            J[1,1] = a21*o**2 + a22*o2 + a22*(o + o2)
            output[i,j,:] = np.dot(J,tmpapp[i,j,:])
    return output.reshape(output.size)

@jit(float64[:](float64,float64[:], int32, int32, int32))
def reaction_update(t,u,Nx,Ny,G):
    uout  = np.zeros(u.size).reshape((Nx,Ny,G))
    tmp = u.reshape((Nx,Ny,G))
    for i in range(Nx):
        for j in range(Ny):
            uout[i,j,:] = full_reaction(t,tmp[i,j,:])
    return uout.reshape(uout.size)
  

def mv(u):
    return u-dt*diffusion_update(0,u,Nx,Ny,dx,dy,G,D,vx,vy,0) - dt*applyJacobian(uold,u)


def mv_jac(u):
    return diffusion_update(0,u,Nx,Ny,dx,dy,G,D,vx,vy,0) + applyJacobian(uold,u)

@jit(float64[:](float64,float64[:]))
def simpleDMD(Yminus,Yplus):
    [U,S,V] = np.linalg.svd(Yminus,full_matrices=False)
    Sinv = np.zeros(S.size)
    if (np.max(np.abs(S))>1e-12):
        Spos = S[S/np.cumsum(S)>1e-9]
        Sinv[0:Spos.size] = 1.0/Spos.copy()
        tmp=np.dot(U.transpose(),Yplus)
        tmp2=np.dot(tmp,V.transpose())
        tmp3=np.dot(tmp2,np.diag(Sinv))
        return np.linalg.eigvals(tmp3)
    else:
        return np.zeros(Yminus.size)

  
Nx = 32
Ny = 32
Lx = 10
Ly = 10
dx = Lx/Nx
dy = Ly/Ny
G = 2
D = np.ones(G)*1e7*0
vx = 0*np.ones(G)
vx[1] = -1e8
vx[0] = 1e8
vy = 0*np.ones(G)
vy[0] = 1e8
vy[1] = -2e8
uold = np.zeros((Nx,Ny,G))+ 1e-6
#uold[:,:,0] = 0.01
uold[int(Nx//4):int(2*Nx//4),int(Ny//4):int(2*Ny//4),0] = 0.01
uold[int(2*Nx//4):int(3*Nx//4),int(2*Ny//4):int(3*Ny//4),1] = 0.01

dRHS = lambda t,uold: diffusion_update(t,uold,Nx,Ny,dx,dy,G,D,vx,vy,0) + reaction_update(t,uold,Nx,Ny,G)
sol = solve_ivp(dRHS, (0,1e-8),uold.flatten(),first_step=1e-14,max_step=1e-9)

plt.pcolor((sol.y[:,-1].reshape((Nx,Ny,G))[:,:,0]))
plt.colorbar()
plt.show()

plt.pcolor((sol.y[:,-1].reshape((Nx,Ny,G))[:,:,1]))
plt.colorbar()
plt.show()
#now do backward Euler linear operator
dts = sol.t[1:] - sol.t[0:-1]
time = 0.0
uold = np.zeros((Nx,Ny,G))+ 1e-6
#uold[:,:,0] = 0.01
uold[int(Nx//4):int(2*Nx//4),int(Ny//4):int(2*Ny//4),0] = 0.01
uold[int(2*Nx//4):int(3*Nx//4),int(2*Ny//4):int(3*Ny//4),1] = 0.01
uold = uold.reshape(Nx*Ny*G)



Nsteps = int(dts.size+1)
times = np.zeros(Nsteps)
sols = np.zeros((Nsteps,Nx*Ny*G))
eigs = np.zeros((Nsteps,Nx*Ny*G), dtype="complex")
evals = np.zeros((Nsteps,Nx*Ny*G))
sols[0,:] = uold.copy()
Yminus = np.zeros((Nsteps,Nx*Ny*G))
Yplus = np.zeros((Nsteps,Nx*Ny*G))
lagDMD = 6
dmd_eigs = np.zeros((Nsteps,Nx*Ny*G), dtype="complex")
for step in range(1,Nsteps):
    dt = dts[step-1]
    idt = 1/dt
    time += dt
    u = uold.copy() 
    mv = lambda u: u-dt*diffusion_update(0,u,Nx,Ny,dx,dy,G,D,vx,vy,0) - dt*applyJacobian(uold,u)
    mv_jac = lambda u, dt: diffusion_update(0,u,Nx,Ny,dx,dy,G,D,vx,vy,0) + applyJacobian(uold,u)
    mv_jac1 = lambda u: diffusion_update(0,u,Nx,Ny,dx,dy,G,D,vx,vy,0) + applyJacobian(uold,u)
    A = spla.LinearOperator((Nx*Ny*G,Nx*Ny*G), matvec=mv, dtype="float64")
    J = spla.LinearOperator((Nx*Ny*G,Nx*Ny*G), matvec=mv_jac1, dtype="float64")
    RHS = dt*diffusion_update(0,uold,Nx,Ny,dx,dy,G,D,vx,vy,0) +dt*reaction_update(0,uold,Nx,Ny,G)
    u, exitCode = spla.lgmres(A, RHS)#x0 = M.dot(RHS))
    print(step,time,dt,exitCode)
    u += uold
    sols[step] = u.copy()
    times[step] = times[step-1]+dt
    Yplus[step] = mv_jac(sols[step] - sols[step-1],dt) #idt*(sols[step] - sols[step-1])-idt*RHS
    Yminus[step] = (sols[step] - sols[step-1])
    tmp_eigs = spla.eigs(J,50)[0]
    eigs[step,0:tmp_eigs.size] = tmp_eigs.copy()
    if (step>=lagDMD):
        tmp_eigs = simpleDMD(Yminus[(step-lagDMD):step,:],Yplus[(step-lagDMD):step,:])
        dmd_eigs[step,0:tmp_eigs.size] = tmp_eigs.copy()
    uold = u.copy()
plt.pcolor(u.reshape((Nx,Ny,G))[:,:,0]-((sol.y[:,-1].reshape((Nx,Ny,G))[:,:,0])))
plt.colorbar()
plt.show()
plt.pcolor(u.reshape((Nx,Ny,G))[:,:,1]-((sol.y[:,-1].reshape((Nx,Ny,G))[:,:,1])))
plt.colorbar()
plt.show()