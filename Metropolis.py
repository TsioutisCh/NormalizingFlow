#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numba import config, njit, threading_layer, prange
import time
import pickle
import lsqfit
import gvar as gv


# In[2]:


@njit 
def diffS(index,xold,xnew,m,o):
    if index == len(xold)-1: #Calculate dS
        dS = 0.5*m*(-2*xnew[0]*(xnew[index] - xold[index]) + xnew[index]**2 - xold[index]**2) +0.5*m*o**2*(xnew[index]**2-xold[index]**2)
    else:
        dS = 0.5*m*(-2*xnew[index+1]*(xnew[index] - xold[index]) + xnew[index]**2 - xold[index]**2) +0.5*m*o**2*(xnew[index]**2-xold[index]**2)
    dS += 0.5*m*(xnew[index]**2-2*xnew[index]*xnew[index-1]-xold[index]**2+2*xold[index]*xold[index-1])
    return dS


# In[3]:


@njit 
def metropolis_sweep(xold,h,m,o,NT):
    accept = 0
    for i in range(NT):#one sweep.
        xnew = xold.copy()
        index = np.random.randint(0,NT) #choose a random index-site from the x-array
        u = np.random.uniform(-h,h) # randomly generate a number [-h,h] from a uniform distribution
        xnew[index] = xold[index] + u
        dS = diffS(index,xold,xnew,m,o)
        prob = np.random.uniform(0,1)
        if prob < min(1,np.exp(-dS)):    
            xold = xnew
            accept+=1
    accept_ratio = accept/NT
    return xold, accept_ratio


# In[4]:


@njit
def autocorrelation(O):
    n = len(O)
    rho = np.zeros(n)
    for t in range(n):
        s0 = 0
        st = 0
        for i in range(n-1-t):
            s0+=O[i]
            st+=O[i+t]
        Oo = s0/(n-t) #<O>o
        Ot = st/(n-t) #<O>t
        sr = 0
        for i in range(n-1-t):
            sr += (O[i]-Oo)*(O[i+t]-Ot)
        rho[t] = sr/(n-t)
    return rho/rho[0]


# In[5]:


def metropolis_conf(NT,N,m,o,h):
    X = np.zeros([N,NT]) 
    for i in range(1,N): X[i] = metropolis_sweep(X[i-1],h,m,o,NT)[0]
    return X


# * Set of parameters

# In[6]:


"""
Nt: The number of paths in the lattice
N: The number of sweeps
m = o: m,o tilde
"""

Nt = np.array([30,36,40,50,60,70,80,100])
m=o=dt = 6/Nt[:]
N = 4_500_000


# In[7]:


#find the acceptance ratio for given Nt
def findH(Nt,N,maxiter):
    k = 0
    h = 0.4 
    H = np.zeros(len(Nt)) 
    for j in range(0,maxiter):
        ratio = np.zeros(N)
        X = np.zeros([N,Nt[k]]) # Array for storing the paths after one sweep
        for i in range(1,N): X[i], ratio[i] = metropolis_sweep(X[i-1],h,m[k],o[k],Nt[k]) 
        target = np.mean(ratio)
        if np.mean(ratio) > 0.79 and np.mean(ratio) < 0.81:
            H[k] = h
            k+=1        
        elif np.mean(ratio) >= 0.81:
            h += 0.1
        elif np.mean(ratio) <= 0.79:
            h -= 0.1
        if k == len(Nt):
            break
    return H


# In[8]:


start = time.time()
H = findH(Nt,N,100)
end = time.time()
print('acceptance radio time:',end - start)
H


# #### Store the simulation in a dictionary, then save it as a pickle file for access.
# 

# In[9]:


if True:
    start = time.time()
    data = {}
    accept = {}
    for k in range(len(Nt)):
        data[k] = metropolis_conf(Nt[k], N, m[k], o[k], H[k])
        print(f"{k+1:3d}/{len(Nt):3d} elapsed={time.time() - start:6.2f} sec")

    end = time.time()
    print('dictionary time',end - start)

    start = time.time()
    with open('data_metropolis.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    end = time.time()
    print('pickle time',end - start)

if True:
    data = pickle.load(open("data_metropolis.pickle", "rb"))


# In[9]:


data = pickle.load(open("data_metropolis.pickle", "rb"))


# Plot $\langle x^2 \rangle$ for each lattice spacing one at a time to
# determine the thermalization time

# In[10]:


for k in range(len(data)):
    plt.plot(np.arange(len(data[k])),np.mean((data[k]*dt[k])**2, axis=1))
    plt.title(f'N = {Nt[k]}')
    plt.xlabel('sweep')
    plt.ylabel('$<x^2>_i$')
    plt.show()


# In[11]:


k = 7
x_squared_mean = np.mean((data[k] * dt[k]) ** 2, axis=1)[:1_000]
metropolis_sweeps = np.arange(len(data[k]))[:1_000]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(metropolis_sweeps[::2], x_squared_mean[::2], marker='o', color='b', s=10)

# Customize plot appearance for scientific use
ax.set_xlabel('Metropolis Sweeps', fontsize=18)  # Increased fontsize
ax.set_ylabel('$<x^2>$', fontsize=18)  # Increased fontsize
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust tick labels for readability
ax.tick_params(axis='both', labelsize=14) 

plt.tight_layout()
plt.savefig('thermalization.png', dpi=300) 


# In[ ]:





# Set ${n_{therm}}$ after inspecting plots. Not necessary in general,
# but here choose same value for all lattice spacings

# In[12]:


ntherm = 10_000


# In[13]:


from scipy.optimize import curve_fit

def fit_func(x, t0):
    return np.exp(-x/t0)


# In[14]:


def autocorrelation(O, tcut=None):
    N = len(O)
    i = np.arange(N)
    cm = O.mean()
    ct = np.fft.fft(O - cm)
    r = np.fft.ifft(ct * ct[(N-i)%N]).real
    return (r/((O-cm)**2).sum())[:tcut]


# In[15]:


def findt0(data, tcut=None):
    rho = autocorrelation(data, tcut=tcut)
    t = np.arange(len(rho))
    popt, pcov = curve_fit(fit_func, t, rho)
    return popt[0]


# In[16]:


start = time.time()
t0 = {k: findt0(((data[k][ntherm:,:]*dt[k])**2).mean(axis=1), tcut=100_000) for k in data}
end = time.time()
print(f" Autocorr. took: {end - start:6.2f} sec")


# In[19]:


rho = {k: autocorrelation(((data[k][ntherm:,:]*dt[k])**2).mean(axis=1), tcut=8_000) for k in data}


# In[23]:


for k in range(len(data)):
    i = np.arange(len(rho[k]))
    plt.plot(i, rho[k], marker=".", ms=1, label=r"$\delta t$ = {:6.3f}".format(dt[k]))
    plt.plot(i, fit_func(i, t0[k]), ls="--", color="k")
plt.ylabel(r"$\rho_{\langle x^2\rangle}(t_{sweep})$", fontsize=18)
plt.xlabel(r"$t_{sweep}$", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('autocorr.png', dpi=300)
plt.show()


# In[20]:


t_skip = {k: 10**int(np.log10(t0[k])+1) for k in t0}
t_skip


# In[21]:


def jackknife(O, N):
    shape = O.shape
    Nsamp = shape[0]
    Nrest = int(np.prod(shape[1:]))
    
    # Reshape `O` into two-dimensional array
    O = O.reshape([Nsamp, Nrest])

    # Fix divisibility with jack-knife bin size
    Nsamp = (Nsamp//N)*N

    # Reshape to three-dimensional, with the second dimension
    # including the samples to subtract
    O = O[:Nsamp, :].reshape([Nsamp//N, N, Nrest])

    # Sum over dataset
    M = O.sum(axis=0).sum(axis=0)

    # Sum over elements to subtract for each jack-knife sample
    R = O.sum(axis=1)

    # New shape is [Nsamp//N, ...]
    new_shape = (R.shape[0],) + shape[1:]

    # Jack-knife bins to return
    bins = np.reshape((M - R)/(Nsamp - N), new_shape)
    return bins


# In[22]:


for k in range(len(data)):
    O = ((data[k][ntherm:,:]*dt[k])**2).mean(axis=1)[ntherm:][::t_skip[k]]
    Ojk = jackknife(O, 1)
    Oave = Ojk.mean(axis=0)
    Oerr = Ojk.std()*np.sqrt(Ojk.shape[0] - 1)
    plt.errorbar(dt[k]**2, Oave, Oerr, color="k", marker="o")
plt.ylim(0.38, 0.62)
plt.xlim(0)
plt.ylabel(r"$\langle x^2\rangle$")
plt.xlabel(r"$\delta t^2$")
plt.show()


# In[23]:


x2_data = []
for k in range(len(data)):
    O = ((data[k][ntherm:,:]*dt[k])**2).mean(axis=1)[ntherm:][::t_skip[k]]
    Ojk = jackknife(O, 1)
    Oave = Ojk.mean(axis=0)
    Oerr = Ojk.std()*np.sqrt(Ojk.shape[0] - 1)
    x2_data.append((dt[k]**2, Oave, Oerr))
x2,ave,err = np.array(x2_data).T


def const(x, p):
    c0, = p
    return c0*x**0

def linear(x, p):
    c0,c1 = p
    return c0 + c1*x

fit_con = lsqfit.nonlinear_fit(data=(x2, gv.gvar(ave, err)), fcn=const, p0=[1,])
fit_lin = lsqfit.nonlinear_fit(data=(x2, gv.gvar(ave, err)), fcn=linear, p0=[1, 1])

plt.errorbar(x2, ave, err, marker="o", ls="")

x = np.linspace(0, 1)
y = const(x, fit_con.p)
plt.plot(x, gv.mean(y), ls=":", color="0.5")
plt.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color="k", alpha=0.3)

x = np.linspace(0, 1)
y = linear(x, fit_lin.p)
plt.plot(x, gv.mean(y), ls="--", color="k")
plt.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color="k", alpha=0.1)

plt.ylim(0.43, 0.56)
plt.xlim(0, 0.025)
plt.ylabel(r"$\langle x^2\rangle$")
plt.xlabel(r"$\delta t^2$")
plt.show()


# In[24]:


# Store fit parameters

data_dict = {
    'x2': x2,
    'ave': ave,
    'err': err,
}

# Specify the file path to save the dictionary
file_path = 'fit_data_metropolis.pkl'

# Save the dictionary to a file using pickle
with open(file_path, 'wb') as file:
    pickle.dump(data_dict, file)

print("Training data saved successfully.")


