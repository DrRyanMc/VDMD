import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Time grid
N = 300 #20000
t = np.logspace(-11,1.65,N) #-11 1.65
t = np.logspace(-11,3.55,N)

dts = np.diff(np.append([0.0],t))

# XS data
data        = np.load('XS_G12.npz')
E           = data['E'] # eV
SigmaT      = data["SigmaT"]
SigmaF      = data["SigmaF"]
SigmaS      = data["SigmaS"]
v           = data["v"]*1000
nu_prompt   = data["nu_prompt"]
nu_delayed  = data["nu_delayed"]
chi_prompt  = data["chi_prompt"]
chi_delayed = data["chi_delayed"]
decay       = data["decay"]
beta_frac   = data["beta_frac"]
#chi_delayed.fill(0.0);

# More data
E_mid            = 0.5*(E[1:]+E[:-1])
dE               = E[1:]-E[:-1]
G                = len(E_mid)
J                = len(decay)
nuSigmaF_prompt  = np.multiply(chi_prompt, nu_prompt*SigmaF)
nuSigmaF_delayed = np.outer(beta_frac, nu_delayed*SigmaF)

# Conversion
eV_J  = 1.60218e-19
mn    = 1.67493E-27 # kg
v_new = np.sqrt(2*E_mid*eV_J/mn)*100.0 # cm/s

# Buckling and leakage XS
R      = 11.7335 # Subcritical
# R      = 11.735 # Supercritical
# R      = 11.74 # Supercritical
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq

# Matrix and RHS source
A = np.zeros([G+J, G+J])
Q = np.zeros(G+J)

# top-left [GxG]: phi --> phi
A[:G,:G] = SigmaS + nuSigmaF_prompt - np.diag(SigmaT+SigmaL)

# top-right [GxJ]: C --> phi
A[:G,G:] = np.multiply(chi_delayed,decay)

# bottom-left [JxG]: phi --> C
A[G:,:G] = nuSigmaF_delayed

# bottom-right [JxJ]: C --> C
A[G:,G:] = -np.diag(decay)

# Multiply with neutron speed
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(v), A[:G,:])

# Eigenmodes: Forward
alpha, vec = eig(AV)
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

# Expansion coefficient
a = np.zeros(G+J)
for n in range(G+J):
    a[n] = np.dot(vec_adj[:G,n],np.diag(1/v).dot(PHI_init[:G]))/\
           (np.dot(vec_adj[:G,n],np.diag(1/v).dot(vec[:G,n])) +\
            np.dot(vec_adj[G:,n],vec[G:,n]))

# Analytical solution
def phi_analytical(t):
    tot = np.zeros(G+J,dtype=complex)
    for n in range(G+J):
        tot[:] += a[n]*vec[:,n]*np.exp(alpha[n]*t)
    return tot

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

# Solve
for n in range(N):
    PHI[:,n] = phi_analytical(t[n])
    PHI_BE[:,n+1] = phi_BE(dts[n],PHI_BE[:,n])
    # Density
    n_tot[n] = sum(np.divide(PHI[:G,n],v))
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
plt.plot(t,n_tot,'k')
plt.plot(t,n_tot_BE,'r--')
plt.xscale('log')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$n$')
plt.grid()
plt.savefig("slab_lin_{}_{}.jpg".format(N,R),bbox_inches='tight' ,dpi=1200)
plt.show()
# loglog
plt.plot(t,n_tot,'k')
plt.plot(t,n_tot_BE,'r--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$n$')
plt.grid()
for a in alpha: 
    a = a.real
    if a > 0:
        plt.axvline(1/a,color='r',alpha=0.5)
    else:
        plt.axvline(-1/a,color='b',alpha=0.5)
plt.ylim(bottom=2.4342013218279325e-15, top=5.50860258832892)
plt.savefig("slab_log_{}_{}.jpg".format(N,R),bbox_inches='tight' ,dpi=1200)
plt.show()

# Plot eigenvalues
plt.plot(alpha.real,alpha.imag,'bo',fillstyle='none')
plt.plot(deigs.real,deigs.imag,'k+',fillstyle='none')
plt.plot(-decay,np.zeros(len(decay)),'rx',label=r"$-\lambda$")
plt.xlabel(r'Re($\alpha$) [/s]')
plt.ylabel(r'Im($\alpha$) [/s]')
plt.legend()
plt.grid()
plt.xlim([-3.0,2.0])
plt.show()

print("analytic",np.sort(alpha.real))
print("DMD", np.sort(deigs))

# =============================================================================
# # Solution with matrix exponential
# # Particular solution
# PHI_p = np.dot(np.linalg.inv(-A),Q)
# 
# # Solve
# for n in range(N):
#     # Flux
#     PHI_h    = np.dot(expm(AV*t[n]),(PHI_init - PHI_p))
#     PHI[:,n] = PHI_h + PHI_p
#     
#     # Density
#     n_tot[n] = sum(np.divide(PHI[:G,n],v))
#     C[:,n]   = PHI[G:,n]
# =============================================================================