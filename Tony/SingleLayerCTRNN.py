import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SingleLayerCTRNN(nn.Module):
    def __init__(self, input_dim=1, h0_data_dim=2, hidden_dim=100, output_dim=1,nonlin=lambda x: torch.relu(torch.tanh(x)),g_init=1.3,store_h=False):
        super().__init__()

        self.nonlin = nonlin
        self.store_h = store_h
        self.input_dim, self.h0_data_dim, self.hidden_dim, self.output_dim = input_dim, h0_data_dim, hidden_dim, output_dim
        self.g_init = g_init

        # Weights and biases for computing initial hidden state from h0_data
        W_h0 = torch.Tensor(h0_data_dim, hidden_dim)
        b_h0 = torch.Tensor(hidden_dim)

        # Weights for input -> hidden transformation
        W_in = torch.Tensor(input_dim, hidden_dim)

        # Weights and biases for recurrent interaction (hidden -> hidden)
        W_rec = torch.Tensor(hidden_dim, hidden_dim)
        b_rec = torch.Tensor(hidden_dim)

        # Weights for hidden -> output transformation
        W_out = torch.Tensor(hidden_dim, output_dim)

        # Parameter setting the timescale of the CTRNN: alpha = dt/tau, where tau is the network time constant
        alpha = torch.Tensor(1)
        
        # Flag all parameters as trainable
        self.W_h0 = nn.Parameter(W_h0)
        self.b_h0 = nn.Parameter(b_h0)

        self.W_in = nn.Parameter(W_in)

        self.W_rec = nn.Parameter(W_rec)
        self.b_rec = nn.Parameter(b_rec)

        self.W_out = nn.Parameter(W_out)

        self.alpha = nn.Parameter(alpha)

        # Initialize weights according to the function defined below
        self.initialize_weights()

        
    def single_step(self, prev_h,curr_i):
        # Implement single timestep for CTRNN: produce new hidden state from previous hidden state and current input
        return ( (1-self.alpha)*prev_h + self.alpha*(self.nonlin(prev_h)@self.W_rec + curr_i@self.W_in + self.b_rec) ).to(device)
    
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

        # Initialize alpha at 0.05
        nn.init.constant_(self.alpha,0.05)

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

        # Calculate initial hidden state
        h = h0_data@self.W_h0 + self.b_h0

        # Produce full hidden state time series
        for t in range(1,T):
            h = torch.cat([h,torch.reshape(self.single_step(h[t-1,:,:], input_ts[t,:,:]),(1,N,self.hidden_dim))],axis=0)
        
        # Comput output time series
        out = self.nonlin(h)@self.W_out #Shape = (T, N, Dim(Out))

        # Return output and hidden time series or just output
        if self.store_h:
          return out,h
        else:
          del h
          return out