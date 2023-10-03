#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numba import config, njit, threading_layer, prange
import time
import pickle
import lsqfit
import gvar as gv


# In[3]:


@njit
def action(x, m, o):
    """
    In:
    x (np.array): Position 

    Out:
    S (vector): Action of given x
    
    """
    n = len(x)
    S = 0 
    for i in range(0,n-1):
        S += 0.5*m*(x[i+1] - x[i])**2 + 0.5*m*o**2*x[i]**2
        
    S+= 0.5*m*(x[0] - x[n-1])**2 + 0.5*m*o**2*x[n-1]**2
    return S


# In[4]:


@njit
def dxS(x, m, o):
    """
    In:
    x (vector): Position (number of sites -> number of paths).

    Out:
    dS (vector): Partial derivative of the action
    
    """    
    
    n = len(x)
    x0 = np.copy(x)
    dS = np.zeros(n)
    dS[0] = -m*(x0[1]-2*x0[0]+x0[-1])+m*o**2*x0[0]
    for i in range(1,n-1):
        dS[i] = -m*(x0[i+1]-2*x0[i]+x0[i-1])+m*o**2*x0[i]
        
    dS[n-1] = -m*(x0[0]-2*x0[n-1]+x0[n-2])+m*o**2*x0[n-1] 
    return dS


# In[5]:


@njit
def leapfrog(x, p, dt, n_steps, m, o):
    """
    Input:
    x (vector): Initial position .
    p (vector): Initial conjugate momentum.
    dt (scalar): Integration time.
    n_steps (scalar): Number of integration steps.

    Output:
    x_new (vector): New position after n_steps.
    p_new (vector): New conjugate momenta after n_steps.
    """

    x_new = np.copy(x)
    p_new = np.copy(p)

    # Half-step update for momenta.
    p_new -= 0.5 * dt * dxS(x_new, m, o)

    for i in range(n_steps - 1):
        # Full-step update for position variables.
        x_new += dt * p_new

        # Full-step update for momenta (except for the last iteration).
        if i < n_steps - 2:
            p_new -= dt * dxS(x_new, m, o)

    # Half-step update for momenta (last iteration).
    p_new -= 0.5 * dt * dxS(x_new, m, o)

    return x_new, p_new


# In[6]:


@njit
def diffH(x_init ,x_fin ,p_init ,p_fin ,m ,o ):
    dH = 0.5*np.dot(p_fin , p_fin) + action(x_fin, m, o) - 0.5*np.dot(p_init , p_init) - action(x_init, m, o)  
    return dH


# In[7]:



def hybrid_monte_carlo(Nt, Nsweeps, m = 1, o = 1, T = 1, t_int = 10**(-1)):
    X = np.zeros([Nsweeps, Nt])
    x_init = np.zeros(Nt) 
    p_init = np.random.normal(0,1,Nt)    
    n_int_steps = int(T/t_int) #integration steps
    c = 0
    for i in range(Nsweeps):
        p_init = np.random.normal(0,1,Nt)
        x_new, p_new = leapfrog(x_init, p_init, t_int, n_int_steps, m, o)
        dH = diffH(x_init, x_new, p_init, p_new ,m ,o )
        if np.random.rand() < np.min([1,np.exp(-dH)]):
            x_init = np.copy(x_new)
            c += 1 
        X[i] = x_init
        
    accept_ratio = c/Nsweeps
    return X , accept_ratio


# In[8]:


"""
Nt: The number of paths in the lattice
N: The number of sweeps
m = o: m,o tilde
"""

Nt = np.array([30,36,40,50,60,70,80,100])
m=o=dt = 6/Nt[:]
N = 4_500_000


# In[ ]:





# Store the simulation in a dictionary, then save it as a pickle file for access.

# In[ ]:


if True:
    start = time.time()
    data = {}
    accept = {}
    for k in range(len(Nt)):
        data[k], accept[k] = hybrid_monte_carlo(Nt[k], N, m[k], o[k], 4)
        print(f"{k+1:3d}/{len(Nt):3d} elapsed={time.time() - start:6.2f} sec")

    end = time.time()
    print('dictionary time',end - start)

    start = time.time()
    with open('data_HMC.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    end = time.time()
    print('pickle time',end - start)

if True:
    data = pickle.load(open("data_HMC.pickle", "rb"))


# In[9]:


data = pickle.load(open("data_HMC.pickle", "rb"))


# Plot $\langle x^2 \rangle$ for each lattice spacing one at a time to
# determine the thermalization time

# In[10]:


for k in range(len(data)):
    plt.plot(np.arange(len(data[k])),np.mean((data[k]*dt[k])**2, axis=1))
    plt.title(f'N = {Nt[k]}')
    plt.xlabel('sweep')
    plt.ylabel('$<x^2>_i$')
    plt.show()


# Set ${n_{therm}}$ after inspecting plots. Not necessary in general,
# but here choose same value for all lattice spacings

# In[11]:


ntherm = 10_000


# In[12]:


from scipy.optimize import curve_fit

def fit_func(x, t0):
    return np.exp(-x/t0)


# Return the autocorrelation time of an observible. Straight-forward
# implementation

# In[13]:


@njit
def autocorrelation_slow(O, tcut=None):
    n = len(O)
    if tcut is None:
        tcut = n
    rho = np.zeros(tcut)
    ave = np.mean(O)
    for t in prange(tcut):
        Op = np.roll(O, shift=-t)
        rho[t] = ((Op-ave)*(O-ave)).mean()
    return rho/rho[0]


# Faster thanks to Fast Fourier Transform trick. Does not jitify due
# to `np.fft()`

# In[14]:


def autocorrelation(O, tcut=None):
    N = len(O)
    i = np.arange(N)
    cm = O.mean()
    ct = np.fft.fft(O - cm)
    r = np.fft.ifft(ct * ct[(N-i)%N]).real
    return (r/((O-cm)**2).sum())[:tcut]


# Fits autocorrelation function to an exponential and returns the
# exponent parameter

# In[15]:


def findt0(data, tcut=None):
    rho = autocorrelation(data, tcut=tcut)
    t = np.arange(len(rho))
    popt, pcov = curve_fit(fit_func, t, rho)
    return popt[0]


# Loop over lattice spacings and compute autocorrelation time

# In[16]:


start = time.time()
t0 = {k: findt0(((data[k][ntherm:,:]*dt[k])**2).mean(axis=1), tcut=10_000) for k in data}
end = time.time()
print(f" Autocorr. took: {end - start:6.2f} sec")


# Get autocorrelation function for each lattice spacing

# In[17]:


rho = {k: autocorrelation(((data[k][ntherm:,:]*dt[k])**2).mean(axis=1), tcut=1_000) for k in data}


# In[18]:


with open('autocorr_hmc.pkl', 'wb') as file:
    pickle.dump(t0, file)


# In[ ]:





# Plot autocorrelation function and fitted curve for each lattice
# spacing

# In[19]:


for k in range(len(data)):
    i = np.arange(len(rho[k]))
    plt.plot(i, rho[k], marker=".", ms=1, label=r"$\delta t$ = {:6.3f}".format(dt[k]))
    plt.plot(i, fit_func(i, t0[k]), ls="--", color="k")
plt.ylabel(r"$\rho_{\langle x^2\rangle}$(t)")
plt.xlabel(r"$t$")
plt.legend()
plt.show()


# Set the correlation length rounding up to the closest power of 10 of
# each value of t0

# In[20]:


t_skip = {k: 10**int(np.log10(t0[k])+1) for k in t0}


# Return jack-knife samples of data set `O` for any jack-knife bin
# size `N`. `O` can have any number of dimensions. The jack-knife
# resampling will be done on the first dimension.

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


# Plot $\langle x^2 \rangle$ for each lattice spacing. Separate by the
# computed correlation length and compute jack-knife errors

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


# Fit $\langle x^2 \rangle = f(\delta t^2)$ to a constant and a linear form

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


# In[ ]:





# In[24]:


# Store fit parameters

data_dict = {
    'x2': x2,
    'ave': ave,
    'err': err,
}

# Specify the file path to save the dictionary
file_path = 'fit_data_hmc.pkl'

# Save the dictionary to a file using pickle
with open(file_path, 'wb') as file:
    pickle.dump(data_dict, file)

print("Training data saved successfully.")

