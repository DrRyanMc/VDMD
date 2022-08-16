#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:50:11 2022

@author: rmcclarr
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
def hide_spines(intx=False,inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""
    font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc', size = 14)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font) 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
           # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font)
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font)
            #ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel(), fontproperties = font)
            ax.set_ylabel(ax.get_ylabel(), fontproperties = font)
            ax.set_title(ax.get_title(), fontproperties = font)
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
def show(nm="",a=0,b=0,show=1):
    hide_spines(a,b)
    if (len(nm)>0):
        plt.savefig(nm,bbox_inches='tight' ,dpi=1200);
    if show:
        plt.show()
    else:
        plt.close()

Nt = 200
np.set_printoptions(precision=20)
dt = 0.1
Yminus = np.zeros((2,Nt))
Yplus = np.zeros((2,Nt))

dts = np.logspace(-3,math.log(3)/math.log(10),Nt)
#dts = 0.1*np.ones(Nt)
trange = np.cumsum(dts)
yval = np.zeros(2)
yval[0]=1
omega = 13/20*np.sqrt(29)
lam = 1/10
#set up matrix
A = np.zeros((2,2))
A[0,1] = 1.0
A[1,0] = -omega**2
A[1,1] = -lam
print("A's eigenvalues", np.linalg.eigvals(A))

w = np.sqrt(omega**2 - lam**2/4)
tau = 2/lam
tplot = np.linspace(0,np.max(trange[0:]),300)
plt.plot(tplot,np.cos(w*tplot)*np.exp(-tplot/tau) + 1/(w*tau)*np.sin(w*tplot)*np.exp(-tplot/tau),'k',alpha=0.75,label="Analytic")
#Backward Euler
for i in range(Nt):
    dt = dts[i]
    update = np.linalg.inv(np.identity(2) - dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval)
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:],Yminus[0,:],'bo', label="Backward Euler")
plt.plot(trange[0:],Yminus[0,:],'b-', alpha=0.25)

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
print("Error =", (-0.05+3.5j)-np.max(deigs), (-0.05-3.5j)-np.min(deigs))


#Crank-Nicolson

yval = np.zeros(2)
yval[0]=1
Yminus = np.zeros((2,Nt))
Yplus = np.zeros((2,Nt))

for i in range(Nt-1):
    dt = dts[i]
    update = np.linalg.inv(np.identity(2) - 0.5*dt*A)
    yold = yval.copy()
    yval = np.dot(update,yval + 0.5*dt*np.dot(A,yval))  
    Yminus[:,i] = 0.5*(yval + yold)
    Yplus[:,i] = (yval-yold)/dt
    
plt.plot(trange[0:],Yminus[0,:],'g^', label="Crank-Nicolson")
plt.plot(trange[0:],Yminus[0,:],'g-', alpha=0.25)
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
print("Error =", (-0.05+3.5j)-np.max(deigs), (-0.05-3.5j)-np.min(deigs))


#BDF-2

yval = np.zeros(2)
yval[0]=1
Yminus = np.zeros((2,Nt))
Yplus = np.zeros((2,Nt))

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
    yval = np.dot(update,4/3*yval-1/3*yold2)  
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-4/3*yold+1/3*yold2)/(2/3*dt)
    
plt.plot(trange[0:],Yminus[0,:],'rv',  label="BDF-2")
plt.plot(trange[0:],Yminus[0,:],'r-', alpha=0.25)
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
print("Error =", (-0.05+3.5j)-np.max(deigs), (-0.05-3.5j)-np.min(deigs))

plt.legend(loc="best")

plt.locator_params(axis='y', nbins=5)
plt.xlabel('t')
plt.ylabel('y(t)')
show("damped_osc.pdf")