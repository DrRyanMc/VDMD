#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:19:05 2021

@author: rmcclarr
"""
import math
import numpy as np

def compute_alpha(psi_input,skip,nsteps,I,G,N,dt):
    """
    Compute eigenvalues/vectors of the operator A in (y^n+1 - y^n) / dt^n = A y^n+1

    Parameters
    ----------
    psi_input : numpy array, psi solutions of size I x G x N x nsteps with thespace, group, angle as the ordering
    skip : integer, number of steps to ignore
    nsteps : integer, number of time steps in the matrix (i.e., number of columns)
    I : integer, number of spatial zones
    G : integer, number of groups
    N : integer, number of angles
    dt : numpy array, 1-D vector of the time steps

    Returns
    -------
    eigsN : numpy array of eigenvalues of the approximate operator
    vsN : numpy array of the eigenvectors
    u : numpy array, U factor from SVD (not sure why this is needed)

    """
    
    
    it = nsteps-1
    
    #need to reshape matrix
    phi_mat = np.zeros((I*G*N,nsteps))
    
    for i in range(nsteps):
        phi_mat[:,i] = np.reshape(psi_input[:,:,:,i],I*G*N)
    #computer time derivative for each step
    phiDT = np.diff(phi_mat,axis=1)/dt
    #print(phiDT.shape)
    [u,s,v] = np.linalg.svd(phi_mat[:,(skip+1):(it+1)],full_matrices=False)
    #print(u.shape,s.shape,v.shape)

    #make diagonal matrix
    #print("Cumulative e-val sum:", (1-np.cumsum(s)/np.sum(s)).tolist())
    #only use important singular values
    spos = s[(1-np.cumsum(s)/np.sum(s)) > 1e-13] 
    mat_size = np.min([I*G*N,len(spos)])
    S = np.zeros((mat_size,mat_size))

    unew = 1.0*u[:,0:mat_size]
    vnew = 1.0*v[0:mat_size,:]

    S[np.diag_indices(mat_size)] = 1.0/spos
    #print(s)
    Atilde = np.dot(np.dot(np.dot(np.matrix(unew).getH(),phiDT),np.matrix(vnew).getH()),S)
    #print("Atilde size =", Atilde.shape)
    [eigsN,vsN] = np.linalg.eig(Atilde)
    return eigsN, vsN,u
