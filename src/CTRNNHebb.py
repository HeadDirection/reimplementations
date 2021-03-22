import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### CTRNN with (short term) Hebbian learning between hidden units
#### https://arxiv.org/pdf/1804.02464.pdf

class CTRNNHebb(nn.Module):
    def __init__(self, input_dim, h0_data_dim, hidden_dim, output_dim,nonlin,g_init=1.3,store_hist=False):
        super().__init__()

        self.nonlin = nonlin
        self.store_hist = store_hist
        self.input_dim, self.h0_data_dim, self.hidden_dim, self.output_dim = input_dim, h0_data_dim, hidden_dim, output_dim
        self.g_init = g_init

        # Weights and biases for computing initial hidden state from h0_data
        W_h0 = torch.Tensor(h0_data_dim, hidden_dim)
        b_h0 = torch.Tensor(hidden_dim)

        # Weights for input -> hidden
        W_in = torch.Tensor(input_dim, hidden_dim)

        # Weights and biases for recurrent interaction (hidden -> hidden)
        W_rec = torch.Tensor(hidden_dim, hidden_dim)
        b_rec = torch.Tensor(hidden_dim)

        # plasticity coefficients for hidden -> hidden
        alpha_rec = torch.Tensor(hidden_dim, hidden_dim)
        hebbtr_rec = torch.Tensor(hidden_dim, hidden_dim)
        hebbtr_rec.requires_grad_(requires_grad=False).to(device)

        # Weights for hidden -> output transformation
        W_out = torch.Tensor(hidden_dim, output_dim)

        # Parameter setting the timescale of the CTRNN: alpha = dt/tau, where tau is the network time constant
        alpha = torch.Tensor(1)

        # Global plasticity parameter
        eta = torch.Tensor(1)
        
        # Flag parameters as trainable
        self.W_h0 = nn.Parameter(W_h0)
        self.b_h0 = nn.Parameter(b_h0)

        self.W_in = nn.Parameter(W_in)

        self.W_rec = nn.Parameter(W_rec)
        self.b_rec = nn.Parameter(b_rec)
        self.alpha_rec = nn.Parameter(alpha_rec)

        self.W_out = nn.Parameter(W_out)

        self.alpha = nn.Parameter(alpha)
        self.eta = nn.Parameter(eta)

        # Initialize weights according to the function defined below
        self.initialize_weights()

        
    def single_step(self, prev_h,curr_i,prev_hebbtr_rec):
        # Implement single timestep for CTRNN with Oja Rule
        # prev_h = previous hidden state (shape N x hidden dim)
        # curr_i = current input (shape N x input dim)
        # prev_hebbtr_rec = previous hebbian trace for recurrent weights (shape N x hidden dim x hidden dim)

        prev_h = prev_h.to(device)
        curr_i = curr_i.to(device)
        prev_hebbtr_rec = prev_hebbtr_rec.to(device)
        N, hidden_dim = prev_h.shape

        prev_r = self.nonlin(prev_h)

        eff_W_rec = (self.W_rec + self.alpha_rec*prev_hebbtr_rec)


        curr_h = ((1-self.alpha)*prev_h 
                  + self.alpha*(torch.bmm(torch.reshape(prev_r,(N,1,hidden_dim)),eff_W_rec).squeeze(1)
                  + curr_i@self.W_in + self.b_rec)).to(device)

        curr_r = self.nonlin(curr_h)

        #curr_r_diags = (torch.cat([torch.diag(curr_r[i,:]).unsqueeze(0) for i in range(N)],axis=0))**2

        new_hebbtr_rec = (
        (1-self.eta)*prev_hebbtr_rec 
        + self.eta*torch.bmm(torch.reshape(prev_r, (N,hidden_dim,1)),torch.reshape(curr_r,(N,1,hidden_dim))) 
        ).to(device)

        return curr_h, new_hebbtr_rec
    
    def initialize_weights(self):
        N = self.hidden_dim
        g = self.g_init  # Parameter controlling variance of initial weights
        sig = g/(N)**0.5

        #nn.init.normal_(self.W_in,0,sig)

        # Initialize input weights at zero
        nn.init.constant_(self.W_in, 0)

        nn.init.normal_(self.W_rec,0,sig)
        nn.init.normal_(self.b_rec,0,sig)
        nn.init.normal_(self.W_out,0,sig)

        nn.init.constant_(self.alpha_rec, 0)

        # Initialize alpha at 0.05
        nn.init.constant_(self.alpha,0.05)

        # Initialize eta at 0.1
        nn.init.constant_(self.eta, 0.1)

        nn.init.normal_(self.W_h0,0,sig)
        nn.init.normal_(self.b_h0,0,sig)

        return None

    
    def forward(self, h0_data, input_ts):
        """
        h0_data has shape (1,N,Dim(h0_data))
        input_ts has shape (T, N, Dim(Input)) where 
        - T is number of timesteps (with spacing dt between timesteps)
        - N is number of samples
        - Dim(Input) is self.input_dim

        Timestep 0 is used exclusively to produce the initial hidden state
        """
        
        T, N = input_ts.size(0), input_ts.size(1)
        hidden_dim = self.hidden_dim

        # Calculate initial hidden state
        h = h0_data@self.W_h0 + self.b_h0

        # initialize hebbian trace
        self.hebbtr_rec = torch.zeros(N,hidden_dim, hidden_dim).to(device)

        # Produce full hidden state time series
        for t in range(1,T):
            new_h, new_hebbtr_rec = self.single_step(h[t-1,:,:], input_ts[t,:,:], self.hebbtr_rec)
            h = torch.cat([h, torch.reshape(new_h, (1,N,hidden_dim))], axis=0)
            self.hebbtr_rec = new_hebbtr_rec
        
        # Comput output time series
        out = (self.nonlin(h)@self.W_out).to(device) #Shape = (T, N, Dim(Out))

        
        # Return output and hidden time series or just output
        if self.store_hist:
          return out,h
        else:
          del h
          return out