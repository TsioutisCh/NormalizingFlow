#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import time
import numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# In[2]:


import base64
import io
import pickle
print(f'TORCH VERSION: {torch.__version__}')
import packaging.version
import matplotlib.pyplot as plt
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
    raise RuntimeError('Torch versions lower than 1.5.0 not supported')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')




def coupling(L, hidden_layer = 3):
    half = L // 2
    nn_size = 128

    layers_s = [nn.Linear(half, nn_size), nn.LeakyReLU()]
    layers_t = [nn.Linear(half, nn_size), nn.LeakyReLU()]

    for _ in range(hidden_layer):
        layers_s += [nn.Linear(nn_size, nn_size), nn.LeakyReLU()]
        layers_t += [nn.Linear(nn_size, nn_size), nn.LeakyReLU()]

    layers_s += [nn.Linear(nn_size, half), nn.Tanh()] #, nn.Tanh()
    layers_t += [nn.Linear(nn_size, half)]

    model_s = nn.Sequential(*layers_s)
    model_t = nn.Sequential(*layers_t)

    return model_s, model_t

def coupling_layers(L,n):
    """
    n: number of layers
    """
    models_s = [coupling(L)[0] for _ in range(n)]
    models_t = [coupling(L)[1] for _ in range(n)]
    
    params = []
    for model in models_s:
        params += list(model.parameters())
    for model in models_t:
        params += list(model.parameters())
    return models_s,models_t, params



def g(u,model_s,model_t,index):  # Affine coupling layer
    
    """
    index:in order to train all the elements 
    """
    L = u.shape[1]
    tensor_size = u.shape[0]
    
    u0 = u[:,(index%2)::2]
    u1 = u[:,((index+1)%2)::2]
    
    s_u0 = model_s(u0)
    t_u0 = model_t(u0)
    
    x0 = u0
    x1 = u1 * torch.exp(s_u0) + t_u0
    
    log_det =  -torch.sum(s_u0, dim = 1)
    
    x0 = x0.reshape([tensor_size,1,L//2])
    x1 = x1.reshape([tensor_size,1,L//2])

    x = [(x0, x1),
         (x1, x0)][index % 2]
    
    return torch.cat(x,dim=1).transpose(1, 2).reshape([tensor_size,L]), log_det


def stack_g(u, models_s,models_t, n):
    '''
    (in)
    u: input tensor
    models: list of models
    n: number of stacking layers
    (out)
    g_u : the result of the last model
    sum_det: the sum of the determinants
    '''
    
    g_stack = []
    log_det_stack = []
    
    g_u = u
    
    for i in range(n):
        model_s = models_s[i]
        model_t = models_t[i]
        g_u, log_det = g(g_u, model_s,model_t,i) 
        log_det_stack.append(log_det.reshape([-1,1]))

    log_det_stack = torch.cat(log_det_stack,dim=1)
    
    log_det_stack = torch.sum(log_det_stack,dim =1) 
    
    
    return g_u, log_det_stack


def logprior(u):
    q = - 0.5*u**2 - torch.log(torch.sqrt(torch.tensor(2 * torch.pi)))
    p_q = torch.sum(q, dim=1)
    return p_q

# KL loss function
def lossKL(p_x, q_x):
    return torch.mean(p_x + q_x) 

def metropolis_check(proposal_distribution, actions, first=True):
    """
    Returns indices of accepted samples in `acc' and acceptance rate in `r_acc'
    """
    acc = []
    r_acc = 0
    j0 = 0
    s0 = actions[0]
    logp0 = proposal_distribution[0]
    if first:
        acc.append(j0)
    for j,s in enumerate(actions[1:],1):
        logp = proposal_distribution[j]
        expon = - (s - s0) - (logp - logp0)
        if expon > 0:
            j0 = j
            s0 = s
            logp0 = logp
            acc.append(j)
            r_acc += 1
        elif np.random.rand() < np.exp(expon.item()):
            j0 = j
            s0 = s
            logp0 = logp
            acc.append(j)
            r_acc += 1
        else:
            acc.append(j0)            
    return acc, r_acc/actions.shape[0]


def action(x, m=1, o=1):
    """
    In:
    x :  (statistics,positions).shape 

    Out:
    S : Action (statistics).shape 
    
    """
    
    x_  = torch.roll(x, shifts=-1, dims=1) # x_ := x[i], x := x[i+1]

    S = 0.5 * m * torch.sum((x - x_)**2, dim = 1) + 0.5 * m * o ** 2 * torch.sum(x_ ** 2, dim = 1)

    return S


# In[4]:


"""
PARAMETERS DEFINITION
"""
L = 100 # number of divisions in time discretization array x (Nt in old code)
dt = 6/L
m,o = dt,dt6 a
tensor_size = 1024 # Statistics dimension in Sequential
n_layers = 6 # number of stacked affine coupling layets
num_epochs = 60_000
racc_freq = 150 # compute acceptance rate every this many epochs
print_freq = 300 # print every this many epochs


# # Training

# In[5]:


#def RealNVP(L, m, o, n_layers, tensor_size, num_epochs):

models_s, models_t, params = coupling_layers(L, n_layers)
optimizer = optim.SGD(params, lr=5e-3)
loss_list = []
accept_list = []
models_dict = {}
# u = torch.randn([tensor_size, L])  # the tensor_size dimension will be hidden (statistic)
start_time = time.time()
for epoch in range(num_epochs+1):
    u = torch.randn([tensor_size, L])   
    x, log_det = stack_g(u, models_s, models_t, n_layers)
    log_det += logprior(u)
    target = action(x, m, o)   
    loss = lossKL(log_det, target)
    loss_list.append((epoch,loss.cpu().detach().numpy()))
    optimizer.zero_grad()
    loss.backward()
    if epoch % racc_freq == 0:
        mcmc_size = 2*tensor_size
        u = torch.randn([mcmc_size, L])   
        x, log_det = stack_g(u, models_s, models_t, n_layers)
        log_det += logprior(u)
        S = action(x, m, o)
        _,r_acc = metropolis_check(log_det.detach(), S.detach())
        accept_list.append((epoch, r_acc))
    if epoch % print_freq == 0:
        print(f'Epoch: {epoch:8d}, Loss: {loss.item():+6.4e}, r: {r_acc:6.3f}')
    optimizer.step()
    if epoch % 2000 == 0:
        models_dict[epoch] = (models_s, models_t)
        print(f'Models saved at epoch {epoch}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
#return models_s, models_t


# Specify the file path to save the models
models_file_path = f'models_L_{L}.pkl'

# Save the models dictionary to a file using pickle
with open(models_file_path, 'wb') as file:
    pickle.dump(models_dict, file)

print("Models saved successfully.")


# # Plot loss and acceptance rate vs training iterations (epochs) 

# In[6]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
x,y = zip(*loss_list)
m0, = ax.plot(x, y, ls="-", lw=0.5, marker="o", ms=2, color=colors[0])
ax2 = ax.twinx()
x,y = zip(*accept_list)
m1, = ax2.plot(x, y, ls="-", marker="s", ms=2, lw=0.5, color=colors[1])
ax.legend((m0,m1), ("Loss","$r_{acc}$"), loc="upper right", frameon=False)
ax2.set_ylim(0, 1)
ax.set_ylabel("Loss")
ax2.set_ylabel("$r_{acc}$")
ax.set_xlabel("Epoch")
ax.set_title(f"L = {L}")
plt.savefig(f"Loss_racc_epoch_L_{L}.png", dpi=300)


# # Compare with HMC (or Metropolis)

# In[7]:


# Generate ensemble using trained model. Do in chuncks

mcmc_size = 4_500_000
mcmc_chunk_size = 10_000 #100_000
initial = None
x_flow = []
for i in range(0, mcmc_size, mcmc_chunk_size):
    u = torch.randn([mcmc_chunk_size, L])   
    x, log_det = stack_g(u, models_s, models_t, n_layers)
    log_det = (log_det+logprior(u)).cpu().detach().numpy()
    S = action(x, m, o).cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    if i != 0:
        S = np.concatenate(([S0], S))
        log_det = np.concatenate(([log_det0], log_det))
        x = np.concatenate(([x0], x))
    acc, r_acc = metropolis_check(log_det, S, first=(i==0))
    x_flow.append(x[acc, :])
    print(f" i = {i:12d}, r = {r_acc:05.3f}")    
    S0,log_det0,x0 = S[acc][-1],log_det[acc][-1],x[acc,:][-1,:]
x_flow = dt*np.array(x_flow).reshape(-1, L)


# In[8]:


## Save x_flow in pickle file

if True:
    start = time.time()
    data = {}
    data = x_flow
    print(f" elapsed={time.time() - start:6.2f} sec")

    end = time.time()
    print('dictionary time',end - start)

    start = time.time()
    with open(f"data_NVP_L_{L}.pickle", 'wb') as f:
        pickle.dump(data, f)
    
    end = time.time()
    print('pickle time',end - start)

if True:
    data = pickle.load(open(f"data_NVP_L_{L}.pickle", "rb"))


# ## Write / Load model parameters

# In[9]:



# Create a dictionary to store the lists
data_dict = {
    'models_s': models_s,
    'models_t': models_t,
    'loss_list': loss_list,
    'accept_list': accept_list,
    'num_epochs': num_epochs,
    'dt': dt,
    'comment': 'models_s models_t loss_list accept_list num_epochs dt'
}

# Specify the file path to save the dictionary
file_path = f'training_data_L_{L}.pkl'

# Save the dictionary to a file using pickle
with open(file_path, 'wb') as file:
    pickle.dump(data_dict, file)

print("Training data saved successfully.")


