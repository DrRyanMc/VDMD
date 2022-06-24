
"""
This file computes the benchmark alpha solutions from the paper


"""

#from ProcessData import NuclearData
import matplotlib.pyplot as plt
from multigroup_sn import *
import math
from mpmath import *

from alphaCalc import *



# In[19]:

def runSlab(cells=100,N=16):
    G = 2
    fuel = 6.696802*2
    refl = 1.126152*2
    L = fuel + refl
    I = int(np.round(cells*L)) #540
    hx = L/I
    Xs = np.linspace(hx/2,L-hx/2,I)
    q = np.ones((I,G))*0
    
    sigma_t = np.ones((I,G))
    nusigma_f = np.zeros((I,G))
    chi = np.zeros((I,G))
    sigma_s = np.zeros((I,G,G))
    count = 0
    for x in Xs:
        #first region
        if x < 1.126152:
            sigma_t[count,0:G] = [2.9865,0.88798]
            sigma_s[count,1,1] = 0.83975
            sigma_s[count,1,0] = 0.04749
            sigma_s[count,0,1] = 0.000336
            sigma_s[count,0,0] = 2.9676
        #second region
        elif x < 6.696802*2 + 1.126152:
            sigma_t[count,0:G] = [2.9727,0.88721]
            sigma_s[count,1,1] = 0.83892
            sigma_s[count,1,0] = 0.04635
            sigma_s[count,0,1] = 0.000767
            sigma_s[count,0,0] = 2.9183
            nusigma_f[count,0:G] = [0.029564*2.5,0.000836*2.5]
            chi[count,0:G] = [0.0,1.0]
        else:
            sigma_t[count,0:G] = [2.9865,0.88798]
            sigma_s[count,1,1] = 0.83975
            sigma_s[count,1,0] = 0.04749
            sigma_s[count,0,1] = 0.000336
            sigma_s[count,0,0] = 2.9676
        count += 1


    plt.plot(Xs,chi)
    plt.plot(Xs,nusigma_f)
    plt.plot(Xs,sigma_t)
    plt.show()
    inv_speed = 1.0

    #N = 196
    MU, W = np.polynomial.legendre.leggauss(N)
    BCs = np.zeros((N,G)) 

    x,k,phi_sol = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs, 
                          tolerance = 1.0e-8,maxits = 4000, LOUD=1 )
    return x,k,phi_sol



def defSlab(cells=100,N=16):
    G = 2
    fuel = 6.696802*2
    refl = 1.126152*2
    L = fuel + refl
    I = int(np.round(cells*L)) #540
    hx = L/I
    Xs = np.linspace(hx/2,L-hx/2,I)
    q = np.ones((I,G))*0
    
    sigma_t = np.ones((I,G))
    nusigma_f = np.zeros((I,G))
    chi = np.zeros((I,G))
    sigma_s = np.zeros((I,G,G))
  
    count = 0
    for x in Xs:
        #first region
        if x < 1.126152:
            sigma_t[count,0:G] = [2.9865,0.88798]
            sigma_s[count,1,1] = 0.83975
            sigma_s[count,1,0] = 0.04749
            sigma_s[count,0,1] = 0.000336
            sigma_s[count,0,0] = 2.9676
        #second region
        elif x < 6.696802*2 + 1.126152:
            sigma_t[count,0:G] = [2.9727,0.88721]
            sigma_s[count,1,1] = 0.83892
            sigma_s[count,1,0] = 0.04635
            sigma_s[count,0,1] = 0.000767
            sigma_s[count,0,0] = 2.9183
            nusigma_f[count,0:G] = [0.029564*2.5,0.000836*2.5]
            chi[count,0:G] = [0.0,1.0]
        else:
            sigma_t[count,0:G] = [2.9865,0.88798]
            sigma_s[count,1,1] = 0.83975
            sigma_s[count,1,0] = 0.04749
            sigma_s[count,0,1] = 0.000336
            sigma_s[count,0,0] = 2.9676
        count += 1
    inv_speed = np.ones(G)
    inv_speed[0] = 1
    #sigma_t[:,0] += -0.8*inv_speed[0]
    inv_speed[1] = .10
    #sigma_t[:,1] += -0.8*inv_speed[1]
    return sigma_t,sigma_s,nusigma_f,chi,inv_speed,q




G = 2
fuel = 6.696802*2
refl = 1.126152*2
L = fuel + refl
cells = 50
N = 8
I = int(np.round(cells*L)) 
hx = L/I
sigma_t,sigma_s,nusigma_f,chi,inv_speed,q = defSlab(cells,N)
MU, W = np.polynomial.legendre.leggauss(N)
psi0 = np.zeros((I,N,G)) + 1e-12
psi0[0,MU>0,1] = 1
psi0[-1,MU<0,1] = 1
numsteps = 300

ts = np.logspace(-2,math.log10(50),numsteps + 1)
dt = np.diff(ts)
print("times =",ts)
print("delta t =",dt)
BCs = np.zeros((N,G)) 
psi0 = np.zeros((I,N,G)) + 1e-12
psi0[0,MU>0,0] = 1
psi0[-1,MU<0,0] = 1
#inv_speed = 1.0

BCs = np.zeros((N,G))
#x,phi,psi = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
#                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 4000, LOUD=0 )


x,k,phi_sol= multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,None, np.random.random((I,G)),
                          tolerance = 1.0e-4,maxits = 40000, LOUD=1 )
print("k =",k)

if (1):
    x,k,phi_sol_alpha, alpha= multigroup_alpha(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,inv_speed,-4.79e-6,-4.77e-6,data_struct=None, phi= np.random.random((I,G)),
                                         tolerance = 1.0e-10,maxits = 40000, LOUD=1, atol=1e-9 )
    print("alpha =",alpha)
x,phi,psi = multigroup_td_var(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,dt, tolerance = 1.0e-8,maxits = 400000, LOUD=1 )
np.savez_compressed("sood_pulse_psi_dt" + str(dt[0]) + "_steps" + str(numsteps), psi=psi, phi=phi, x=x)

#get eigenvalues
step = 0
included = numsteps+1
eigsN, vsN,u = compute_alpha(psi[:,:,:,step:(step+included+1)],0,included,I,G,N,dt)
print(vsN.shape,u.shape)
print(eigsN[ np.abs(np.imag(eigsN)) < 1e-6])

assert 0
psi0 = np.random.uniform(high=1,low=0,size=(I,N,G)) + 1e-12
x,phi2,psi2 = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 4000, LOUD=0 )
np.savez_compressed("sood_rand_psi_dt" + str(dt[0]) + "_steps" + str(numsteps), psi=psi, phi=phi, x=x)

numsteps = 10000
dt = 1.0e-2
psi0 = np.zeros((I,N,G)) + 1e-12
psi0[0,MU>0,1] = 1
psi0[-1,MU<0,1] = 1
x,phi,psi = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 4000, LOUD=0 )
np.savez_compressed("sood_pulse_psi_dt" + str(dt) + "_steps" + str(numsteps), psi=psi, phi=phi, x=x)


psi0 = np.random.uniform(high=1,low=0,size=(I,N,G)) + 1e-12
x,phi2,psi2 = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 4000, LOUD=0 )
np.savez_compressed("sood_rand_psi_dt" + str(dt) + "_steps" + str(numsteps), psi=psi, phi=phi, x=x)



