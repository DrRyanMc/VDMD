import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Time grid
N = 600
t = np.logspace(-10,3.,N)

dts = np.diff(np.append([0.0],t))

# XS data
data       = np.load('SHEM-361.npz')
SigmaT     = data['SigmaT']
nuSigmaF   = data['nuSigmaF']
SigmaS     = data['SigmaS']
SigmaF     = data['SigmaF']
v          = data['v']
nuSigmaF_p = data['nuSigmaF_p']
chi_d      = data['chi_d']
lamd       = data['lamd']
nu_d       = data['nu_d']
E          = data['E']

# More data
G = len(SigmaT)
J = len(lamd)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

# Buckling and leakage XS
R      = 81.0 # Sub
R      = 81.5
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaT += SigmaL

# Matrix and RHS source
A = np.zeros([G+J, G+J])
Q = np.zeros(G+J)

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
AV[:G,:] = np.dot(np.diag(v), A[:G,:])
alpha,vec = np.linalg.eig(AV)
idx = alpha.argsort()[::-1]   
alpha = alpha[idx]
vec = vec[:,idx]

# Eigenmodes: Adjoint
A_adj        = np.transpose(A)
AV_adj       = np.copy(A_adj)
AV_adj[:G,:] = np.dot(np.diag(v), A_adj[:G,:])
alpha_adj, vec_adj = eig(AV_adj)
idx = alpha_adj.argsort()[::-1]   
alpha = alpha_adj[idx]
vec_adj = vec_adj[:,idx]

# Initial condition
PHI_init = np.zeros(G+J)
#PHI_init[0:G] = np.random.rand(G)*v
PHI_init[G-1] = v[-1]

# Backward Euler solution
def phi_BE(dt,phi0):
    tot = np.linalg.solve(np.identity(G+J,dtype=complex) - dt*AV,phi0)
    #tot = np.linalg.solve(np.identity(G+J,dtype=complex) - 0.5*dt*AV,np.dot(np.identity(G+J,dtype=complex) + 0.5*dt*AV,phi0)) #CN
    #tot = np.dot(expm(AV*dt),phi0)
    return tot

# Allocate solution
PHI   = np.zeros([G+J,N])
C     = np.zeros((J,N))
n_tot = np.zeros(N)

PHI_BE = np.zeros([G+J,N+1])
C_BE     = np.zeros((J,N))
n_tot_BE = np.zeros(N)
PHI_BE[:,0] = PHI_init.copy()

# Analytical particular solution
PHI_p = np.dot(np.linalg.inv(-A),Q)

# Analytical solution
for n in range(N):
    # Flux
    PHI_h    = np.dot(expm(AV*t[n]),(PHI_init - PHI_p))
    PHI[:,n] = PHI_h + PHI_p
    PHI_BE[:,n+1] = phi_BE(dts[n],PHI_BE[:,n])    
    
    # Density
    n_tot[n] = sum(np.divide(PHI[:G,n],v))
    C[:,n]   = PHI[G:,n]
    n_tot_BE[n] = sum(np.divide(PHI_BE[:G,n+1],v))
    C_BE[:,n]   = PHI_BE[G:,n+1]

# Save solution    
np.savez('analytical.npz',t=t,phi=PHI[:G,:],n_tot=n_tot,C=C,alpha=alpha)

#DMD
#compute the derivative

dPHI = np.diff(PHI_BE,axis=1)/dts #difference divided by dt

#take SVD of solution from time n+1
[U,S,V] = svd(PHI_BE[:,1:],full_matrices=False)
Sinv = np.zeros(S.size)
Spos = S[S/np.cumsum(S)>1e-14]
Sinv[0:Spos.size] = 1.0/Spos.copy()
tmp=np.dot(U.transpose(),dPHI)
tmp2=np.dot(tmp,V.transpose())
tmp3=np.dot(tmp2,np.diag(Sinv))
deigs = np.linalg.eigvals(tmp3)

# Plot solution
for a in alpha: 
    a = a.real
    if a > 0:
        plt.axvline(1/a,color='r',alpha=0.5)
    else:
        plt.axvline(-1/a,color='b',alpha=0.5)
plt.plot(t,n_tot,'k')
plt.plot(t,n_tot_BE,'r--')
plt.xscale('log')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$n$')
plt.grid()
plt.show()

# loglog
for a in alpha: 
    a = a.real
    if a > 0:
        plt.axvline(1/a,color='r',alpha=0.5)
    else:
        plt.axvline(-1/a,color='b',alpha=0.5)
plt.plot(t,n_tot,'k')
plt.plot(t,n_tot_BE,'r--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$n$')
plt.grid()
plt.show()

# Plot eigenvalues
for j in range(J):
    if j == 0:
        plt.axvline(-lamd[j],color='g',alpha=0.5,label=r"$-\lambda$")
    else:
        plt.axvline(-lamd[j],color='g',alpha=0.5)
plt.plot(alpha.real,alpha.imag,'bo',fillstyle='none',label='Ref.')
plt.plot(deigs.real,deigs.imag,'rx',fillstyle='none',label='VDMD')
plt.xlabel(r'Re($\alpha$) [/s]')
plt.ylabel(r'Im($\alpha$) [/s]')
plt.legend()
plt.grid()
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

print("analytic",np.sort(alpha.real))
print("DMD", np.sort(deigs))