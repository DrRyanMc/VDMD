import numpy as np
import matplotlib.pyplot as plt

data       = np.load('SHEM-361.npz')
SigmaT     = data['SigmaT']
nuSigmaF   = data['nuSigmaF']
SigmaS     = data['SigmaS']
SigmaF     = data['SigmaF']
speed      = data['v']
nuSigmaF_p = data['nuSigmaF_p']
chi_d      = data['chi_d']
lamd       = data['lamd']
nu_d       = data['nu_d']
E          = data['E']

G = len(SigmaT)
J = len(lamd)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

#===============================================================================
# k-eigenvalue
#===============================================================================

A = np.dot(np.linalg.inv(np.diag(SigmaT) - SigmaS),nuSigmaF)
w,v = np.linalg.eig(A)
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]
k = w[0]
phi_k = v[:,0]
if phi_k[0] < 0.0:
    phi_k *= -1
phi_k[:] /= np.sum(phi_k[:])


#===============================================================================
# alpha-eigenvalue
#===============================================================================

A = np.zeros([G+J,G+J])

# Top-left [GxG]: phi --> phi
A[:G,:G] = SigmaS + nuSigmaF_p - np.diag(SigmaT)

# Top-right [GxJ]: C --> phi
A[:G,G:] = np.multiply(chi_d,lamd)

# Bottom-left [JxG]: phi --> C
A[G:,:G] = np.multiply(nu_d,SigmaF)

# bottom-right [JxJ]: C --> C
A[G:,G:] = -np.diag(lamd)

# Multiply with neutron speed
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(speed), A[:G,:])
w,v = np.linalg.eig(AV)
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]
alpha = w[0]
phi_alpha = v[:,0]
if phi_alpha[0] < 0.0:
    phi_alpha *= -1
phi_alpha[:G] /= np.sum(phi_alpha[:G])

plt.plot(E_mid, phi_alpha[:G]*E_mid/dE,'b-')
plt.plot(E_mid, phi_k*E_mid/dE,'r--')
plt.xscale('log')
plt.show()

plt.plot(w.real,w.imag,'o')
plt.show()