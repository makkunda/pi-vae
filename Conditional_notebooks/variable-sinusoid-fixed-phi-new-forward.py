# from 1dgp
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

#from VAE,piVAE
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn.parameter import Parameter
from tqdm import tqdm, trange
import cmdstanpy
import pandas as pd
import pickle
import math

#for logging

import wandb

# for debugging

import ipdb

wandb.init(
    # set the wandb project where this run will be logged
    project="pi-vae",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.00003,
    "dataset": "2-parameter-sinosoid",
    "epochs": 5000,
    },
    name="variable-sinusoid-fixed-phi-full-batch-new-forward"
)

def parameterized_sin(x,a=1,b=2,c=3):
    b = np.random.normal(0, 1)
    c = np.random.normal(0, 1)
    return a*np.sin(b*x+c) + 0.1*np.random.normal(0, 1, size=x.shape)

class Sin1D(Dataset):
    def __init__(self, dataPoints=100, samples=10000, ingrid=False, x_lim = 3,
                        seed=np.random.randint(20),ls = 0.1, nu=2.5):
        self.dataPoints = dataPoints
        self.samples = samples
        self.ingrid = ingrid
        self.x_lim = x_lim
        self.seed = seed
        self.Max_Points = 2 * dataPoints
        self.ls = ls
        self.nu = nu
        np.random.seed(self.seed)
        self.evalPoints, self.data = self.__simulatedata__()
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx=0):
        return(self.evalPoints[:,idx], self.data[:,idx])


    def __simulatedata__(self):
        
        if (self.ingrid):
            X_ = np.linspace(-self.x_lim, self.x_lim, self.dataPoints)
            y_samples = parameterized_sin(X_.repeat(self.samples).reshape(X_.shape[0],self.samples))
            # print(X_.shape, y_samples.shape)
            return (X_.repeat(self.samples).reshape(X_.shape[0],self.samples) ,
                        y_samples)
        else:
            X_ = np.linspace(-self.x_lim, self.x_lim, self.Max_Points)
            X_ = np.random.choice(X_, (self.dataPoints,self.samples))
            X_.sort(axis=0)
            y_samples = np.zeros((self.dataPoints,self.samples))
            for idx in range(self.samples):
                x_ = X_[:,idx]
                y_samples[:,idx] =  parameterized_sin(x_[:]).reshape(self.dataPoints,)
            # print(X_.shape, y_samples.shape)
            return (X_, y_samples)
        
def visualize_1D_sin():
    dataset =Sin1D(dataPoints=100, samples=10000, ls=0.1, x_lim=3)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for no, dt in enumerate(dataloader):
        ax.plot(dt[0].reshape(-1,1), dt[1].reshape(-1,1), marker='o', markersize=3)
        if no > 9: break
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f(x)$')
    ax.set_title('10 different function realizations at fixed 100 points\n'
    'sampled from differently parameterized sin(ax+b) functions')
#     fig_image = wandb.Image(fig)
    wandb.log({"data visualization": fig})

visualize_1D_sin()

class PHI(nn.Module):
    '''
    Implementation of feature transformation layer with RBF layer.
    We assume here that alpha is constant for all basis.
    Shape:
        - Input: (N, n_evals, in_features) N is batches
        - Output: (N, n_evals, out_dims), out_dims is a parameter
    Parameters:
        - in_features: number of input dimension for each eval point
        - alpha - trainable parameter controls width. Default is 1.0
        - n_centers - number of points to be used as centers in rbf/matern
        layers. centers are trainable, default is 100
        - hidden_dim1: hidden dimension for 1st layer. Default is 20
        - hidden_dim2: hidden dimension for 2nd layer. Default is 20
        - out_dims: output features to construct. Default is 100
    Examples:
        >>> a1 = PHI(256)
        >>> x = torch.randn(1,256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = 1.0, n_centers = 10, 
                    hidden_dim1 = 20, hidden_dim2 = 20, out_dims = 100):
        '''
        Initialization.
        INPUT:
            - in_features: number of input dimension for each eval point
            - alpha: trainable parameter
            alpha is initialized with 1.0 value by default
            - n_centers: number of points to be used as centers in rbf/matern
            layers. centers are trainable, default is 100
            - hidden_dim1: hidden dimension for 1st layer. Default is 20
            - hidden_dim2: hidden dimension for 2nd layer. Default is 20
            - out_dims: hidden dimension for 2nd layer. Default is 100
        '''
        super(PHI,self).__init__()
        self.in_features = in_features

        # initialize alpha
       # self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        #self.alpha.requiresGrad = True # set requiresGrad to true!
        # centers
        
        n_centers = out_dims
        
        self.centers = Parameter(torch.randn(n_centers, in_features)) # create a tensor out of centers
        self.centers.requiresGrad = False # set requiresGrad to false!, so fixed centres chosen randomly
        # linear layers
#         self.linear1 = nn.Linear(n_centers, hidden_dim1)
#         self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.out = nn.Linear(n_centers, out_dims)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        rbf = torch.exp(-1 * torch.cdist(x, self.centers).pow(2))
#         hidden1 = torch.tanh(self.linear1(rbf))
#         hidden2 = torch.tanh(self.linear2(hidden1))
#         out = self.out(rbf)
        out = rbf
        return out

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.sd = nn.Linear(hidden_dim2, z_dim)
    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        z_mu = self.mu(hidden2)
        # z_mu is of shape [batch_size, z_dim]
        z_sd = self.sd(hidden2)
        # z_sd is of shape [batch_size, z_dim]
        return z_mu, z_sd

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, input_dim)
    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        pred = self.out(hidden2)
        # pred is of shape [batch_size, input_dim]
        return pred

class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim1, hidden_dim2, input_dim)

    def reparameterize(self, z_mu, z_sd):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if self.training:
            # sample from the distribution having latent parameters z_mu, z_sd
            # reparameterize
            std = torch.exp(z_sd / 2)
            eps = torch.randn_like(std)
            return (eps.mul(std).add_(z_mu))
        else:
            return z_mu


    def forward(self, x):
        # encode
        z_mu, z_sd = self.encoder(x)
        # reparameterize
        x_sample = self.reparameterize(z_mu, z_sd)
        # decode
        generated_x = self.decoder(x_sample)
        return generated_x, z_mu,z_sd

def calculate_loss_VAE(x, reconstructed_x, mean, log_sd):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_sd - mean.pow(2) - log_sd.exp())
    return RCL + KLD

class PIVAE(nn.Module):
    '''
    Implementation of PIVAE with feature transformation layer (RBF layer).
    Shape:
        - Input: (N, n_evals, in_features) N is batches
        - Output: (N, n_evals, 1), currently we have 1D output only
    Parameters:
        - in_features: number of input dimension for each eval point
        - alpha - trainable parameter controls width. Default is 1.0
        - n_centers - number of points to be used as centers in rbf/matern
        layers. centers are trainable, default is 100
        - dim1: hidden dimension for 1st transformation layer. Default is 20
        - dim2: hidden dimension for 2nd layer. Default is 20
        - out_dims: output features to construct (size of beta and VAE). 
        Default is 100
        - hidden_dim1 - hidden dimensions for 1st layer VAE. Default is 128
        - hidden_dim2 - hidden dimensions for 1st layer VAE. Default is 64
        - z_dim - latent dimension for VAE. Default is 20
        - batch_size - batch_size for training. For now set same as n_samples
    Examples:
        >>> a1 = PHI(256)
        >>> x = torch.randn(1,256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = 1.0, n_centers = 10, dim1 = 20, 
                    dim2 = 20, out_dims = 100, hidden_dim1 = 128, 
                    hidden_dim2 = 64, z_dim = 20, batch_size = 10000):
        super(PIVAE, self).__init__()
        self.out_dims = out_dims
        self.batch_size = batch_size
        self.phi = PHI(in_features, alpha=alpha, n_centers=n_centers, 
                        hidden_dim1=dim1, hidden_dim2=dim2, out_dims=out_dims)
#         self.betas = nn.ModuleList()
#         for _ in range(self.batch_size):
#             self.betas.append(nn.Linear(out_dims, 1))
        self.betas = Parameter(torch.randn(batch_size, out_dims)) # create a beta matrix
        self.betas.requiresGrad = True
        
        self.vae = VAE(input_dim=out_dims, hidden_dim1=hidden_dim1, 
                        hidden_dim2=hidden_dim2, latent_dim=z_dim)
    
    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''        
        phi_x = self.phi(x)
        
        y1 = torch.einsum("bo,bno->bn",[self.betas,phi_x])
        
        beta_vae, z_mu, z_sd = self.vae(self.betas)
        
        y2 = torch.einsum("bo,bno->bn",[beta_vae,phi_x])
        
#         ipdb.set_trace()

        return y1, y2, z_mu, z_sd
    
def calculate_loss(target, reconstructed1, reconstructed2, mean, log_var):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed1, target, reduction='sum') + \
                F.mse_loss(reconstructed2, target, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL + KLD

# sampling points to evaluate functions values
x_inf = np.linspace(-3.5,3.5,100).reshape(-1,1)
e_n = 0.1 * np.random.randn(100).reshape(-1,1)

a=1
b = np.random.normal(0, 1)
c = np.random.normal(0, 1)
y_ = a*np.sin(b*x_inf+c)
y_inf = y_ + e_n
idx = (x_inf>=-3) * (x_inf<=3)
ll_idx = np.where(idx)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_inf, y_, color='blue', alpha=0.5)
ax.scatter(x_inf[ll_idx], y_inf[ll_idx], marker='+', color='red', alpha=0.5, s=100)
ax.set_xlabel('$x$')
ax.set_ylabel('$y=f(x)$')
image = wandb.Image(fig)
wandb.log({"posterior": image})

sm = cmdstanpy.CmdStanModel(stan_file='../notebooks/pivae.stan')

sm1 = cmdstanpy.CmdStanModel(stan_file='../notebooks/prior_predictive.stan')

def train_piVAE():
    # Just showing how to use piVAE to learn priors

    ###### intializing data and model parameters
    n_samples = 109
    in_features = 1
    n_evals = 103
    n_centers = math.ceil(n_evals/2)
    alpha = 1.0
    dim1 = 20
    dim2 = 20
    hidden_dims1 = 16
    hidden_dims2 = 8
    z_dim = 5
    out_dims = 104
    batch_size = 97

    ###### creating data, model and optimizer
    train_ds = Sin1D(dataPoints=n_evals, samples=n_samples, ls=0.1)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False,drop_last=True)
    
    val_ds = Sin1D(dataPoints=n_evals, samples=n_samples)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,drop_last=True)
    
    model = PIVAE(in_features=in_features, alpha=alpha, n_centers=n_centers,
                     dim1=dim1, dim2=dim2, out_dims=out_dims, 
                     hidden_dim1=hidden_dims1, hidden_dim2=hidden_dims2, 
                     z_dim=z_dim, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = model.to(device)
    
    epochs = 50000
    print(device)
    ###### running for 5000 epochs
    t = trange(epochs)
    for e in t:
        # set training mode
        
        if e == 0 or e%100 == 0: # running inference as a test every 100 epoch
            phi = model.phi
            
            vae = model.vae
            vae_decoder = vae.decoder
            
            stan_data = {'p': 5, 
                 'p1': 16,
                 'p2': 8,
                 'n': 100,
                 'W1': vae_decoder.linear1.weight.T.cpu().detach().numpy(),
                 'B1': vae_decoder.linear1.bias.T.cpu().detach().numpy(),
                 'W2': vae_decoder.linear2.weight.T.cpu().detach().numpy(),
                 'B2': vae_decoder.linear2.bias.T.cpu().detach().numpy(),
                 'W3': vae_decoder.out.weight.T.cpu().detach().numpy(),
                 'B3': vae_decoder.out.bias.T.cpu().detach().numpy(),
                 'beta_dim' : out_dims,
                 'phi_x' : phi(torch.tensor(x_inf).float().to(device)).cpu().detach().numpy(),
                 'y': y_inf.reshape(100,),
                 'll_len' : ll_idx[0].shape[0],
                 'll_idxs' : ll_idx[0]}
            
            fit = sm.sample(data=stan_data, iter_sampling=2000, iter_warmup=500, chains=4)
            
            fit1 = sm1.sample(data=stan_data, iter_sampling=2000, iter_warmup=500, chains=4)
            
            out = fit.stan_variables()

            df = pd.DataFrame(out['y2'])
            
            out1 = fit1.stan_variables()

            df1 = pd.DataFrame(out1['y2'])
            
            datapoints = x_inf
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(datapoints, y_, color='black', label='True')
            ax.scatter(datapoints[ll_idx], y_inf[ll_idx], s=46,label = 'Observations')
            ax.fill_between(datapoints.reshape(datapoints.shape[0]), df.quantile(0.025).to_numpy(), df.quantile(0.975).to_numpy(),
                                facecolor="blue",
                                color='blue', 
                                alpha=0.2, label = '95% Credible Interval') 
            ax.plot(datapoints, df.mean().to_numpy().reshape(-1,1), color='red', alpha=0.7, label = 'Posterior mean')
            plt.xlim(-10, 10)
            plt.ylim(-2, 2)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y=f(x)$')
            ax.set_title('Inference fit')
            ax.legend()
            image = wandb.Image(fig)
            wandb.log({"inference fit": image})
            
            df2 = df1.to_numpy()
            
            df3_0 = df2[0]
            df3_1 = df2[1]
            df3_2 = df2[2]
            df3_3 = df2[3]
            df3_4 = df2[4]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(datapoints, df3_0.reshape(-1,1), alpha=0.7, label = 'Posterior 1')
            ax.plot(datapoints, df3_1.reshape(-1,1), alpha=0.7, label = 'Posterior 2')
            ax.plot(datapoints, df3_2.reshape(-1,1), alpha=0.7, label = 'Posterior 3')
            ax.plot(datapoints, df3_3.reshape(-1,1), alpha=0.7, label = 'Posterior 4')
            ax.plot(datapoints, df3_4.reshape(-1,1), alpha=0.7, label = 'Posterior 5')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y=f(x)$')
            ax.set_title('Draws from pi-vae prior')
            ax.legend()
            image = wandb.Image(fig)
            wandb.log({"Draws from pi-vae prior": image})
            
        model.train()
        total_loss = 0
        for i,x in enumerate(train_dl):
            target = x[1].float().to(device)
            # target = target.view(target.shape[0], target.shape[1], 1)
            x = x[0].float().to(device)
            x = x.view(x.shape[0], x.shape[1], 1)
            optimizer.zero_grad()   # zero the gradient buffers
            y1, y2, z_mu, z_sd = model(x) # fwd pass
            loss = calculate_loss(target, y1, y2, z_mu, z_sd) # loss cal
            loss.backward() # bck pass
            total_loss += loss.item() 
            optimizer.step() # update the weights
        loss_logging = total_loss/(n_evals*n_samples)
        wandb.log({"train_loss": loss_logging})
        t.set_description(f'Loss is {total_loss/(n_evals*n_samples):.3}')
        
        total_val_loss = 0
        for i,x in enumerate(val_dl):
            target = x[1].float().to(device)
            # target = target.view(target.shape[0], target.shape[1], 1)
            x = x[0].float().to(device)
            x = x.view(x.shape[0], x.shape[1], 1)
            y1, y2, z_mu, z_sd = model(x) # fwd pass
            loss = calculate_loss(target, y1, y2, z_mu, z_sd) # loss cal
            total_val_loss += loss.item() 
        loss_logging_val = total_val_loss/(n_evals*n_samples)
        wandb.log({"val_loss": loss_logging_val})
        t.set_description(f'Val Loss is {total_loss/(n_evals*n_samples):.3}')
        
    
    return model

model = train_piVAE()

pickle.dump(model, open("variable-sinusoid-model-fixed-center.pkl", "wb") )
