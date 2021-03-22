import numpy as np
import torch
from SingleLayerCTRNN import SingleLayerCTRNN
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hdts = np.load('angs_smooth.npy')
avt = hdts[0][1:-1]
dt = np.mean(np.diff(avt))
av = (hdts[1][2:]-hdts[1][:-2])/(2*dt)
av_normalized = av/np.mean(np.abs(av))
hdts_avt = hdts[1][1:-1]

t_steps = 42000
samp_length = 700
samp_ct = t_steps//samp_length

all_ts = np.vstack([avt, np.sin(hdts_avt), np.cos(hdts_avt), av_normalized])
batches = np.transpose(np.reshape(all_ts[:,:t_steps], (4, samp_ct, samp_length)), (2,1,0))

input_h0_data_samps = []
input_ts_samps = []
output_ts_samps = []


for i in range(samp_ct):
    sample = batches[:,i,:]
    output_samp = sample[:,1:3]
    input_samp = sample[:,3]
    input_h0_data = sample[0,1:3]
    
    input_h0_data_samps += [input_h0_data]
    input_ts_samps += [input_samp]
    output_ts_samps += [output_samp]

fin_h0_dat = torch.from_numpy(np.stack(input_h0_data_samps, axis=0)).float().unsqueeze(0).to(device)
fin_input = torch.from_numpy(np.stack(input_ts_samps, axis=1)).float().unsqueeze(2).to(device)
fin_output = torch.from_numpy(np.stack(output_ts_samps, axis=1)).float().to(device)

nonlin = lambda x : torch.maximum(torch.tensor([0]).to(device),nn.Tanh()(x))
model = SingleLayerCTRNN(input_dim=1, h0_data_dim = 2, hidden_dim=100, output_dim=2, nonlin = nonlin,store_h=False).cuda()

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = .005, weight_decay=1e-6)

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    pred = model(fin_h0_dat, fin_input)
    loss = criterion(pred, fin_output)

    loss.backward()
    optimizer.step()
    
    # if (epoch+1) % 10 == 0:
    print ('Epoch [{}/{}], Loss: {:.4f}' 
            .format(epoch+1, num_epochs, float(loss)))
