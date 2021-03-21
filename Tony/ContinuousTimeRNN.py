import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class ContinuousTimeRNN(nn.Module):
    def __init__(self, init_dim=2, input_dim=1, hidden_dim=100, output_dim=1, alpha=0.1):
        """
        output_dim=1 instead of 2 because predicting raw radians instead of sin() and cos()
        """
        super(ContinuousTimeRNN, self).__init__() 

        self.Din = input_dim 
        self.H = hidden_dim 
        self.Dout = output_dim 
        self.alpha = alpha

        self.fc = nn.Linear(init_dim, hidden_dim)
        nn.init.normal_(self.fc.weight, mean=0, std=(0.1/math.sqrt(self.fc.weight.size(0))))

        # TODO: Are these weight init good?
        self.W_in = nn.Parameter( nn.init.normal_(torch.randn((input_dim, hidden_dim)), mean=0, std=(0.1/math.sqrt(input_dim))) )
        self.W_rec = nn.Parameter( nn.init.normal_(torch.randn((hidden_dim, hidden_dim)), mean=0, std=(0.1/math.sqrt(hidden_dim))) ) 
        self.W_out = nn.Parameter( nn.init.normal_(torch.randn((hidden_dim, output_dim)), mean=0, std=(0.1/math.sqrt(hidden_dim))) )
        self.bias = nn.Parameter( torch.zeros(hidden_dim) )

    def step_forward(self, x, prevH):
        """
        x (N, D) - examples at a single timestep
            prevH (N, H) - hidden unit activations at the previous timestep
        """
        N = x.size(0)
        activatedPrevH = torch.relu(torch.tanh(prevH))
        biasMatrix = torch.cat(N * [torch.unsqueeze(self.bias, dim=0)])

        term1 = (1 - self.alpha) * prevH
        term2 = self.alpha * ( (activatedPrevH @ self.W_rec) + (x @ self.W_in) + biasMatrix )
        
        return term1 + term2 # TODO No noise yet

    def forward(self, initdir, velocities):
        """
        initdir - (N, initdim) initial direction (for us it's [sin(theta0), cos(theta0)])
        velocities (T, N, D) - input vector for each example at every timestep 

        returns (T, N, Dout) - predictions for each example at each timesteps
        """
        T, N, D = velocities.shape

        h = torch.zeros((T, N, self.H))
        h0 = self.fc(initdir)

        for t in range(T):
            if t == 0:
                h[t, :, :] = self.step_forward(velocities[t, :, :], h0)
            else:
                h[t, :, :] = self.step_forward(velocities[t, :, :], h[t-1])

        out = h @ self.W_out
        return out

