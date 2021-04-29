import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN
from simple_data_gen import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SingleLayerCTRNN(store_h=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    model.load_state_dict(torch.load('activations_fakedata.pt', map_location=torch.device(device)))

    inputs = None 
    outputs = None
    hidden_states = None
    for i in range(10):
        train_data_size = 100
        train_data_len = 700
        dt = .025
        initdir, input, output = gen_batch(train_data_size,train_data_len,dt,25,[],7, [0.0])
        pred, h = model(initdir, input)
        loss = criterion(pred, output)
        print(f"Loss on test data: {loss.item():.4f}")

        np_input = np.array(input.cpu().detach().numpy())
        np_output = np.array(output.cpu().detach().numpy())
        np_h = np.array(h.cpu().detach().numpy())
        if inputs is None:
            inputs = np_input 
            outputs = np_output
            hidden_states = np_h
        else:
            inputs = np.concatenate((inputs, np_input), axis=1)
            outputs = np.concatenate((outputs, np_output), axis=1)
            hidden_states = np.concatenate((hidden_states, np_h), axis=1)

    train_data_size = 100
    train_data_len = 700
    # dt = .025
    dt = 0.0025
    initdir, input, output = gen_batch(train_data_size,train_data_len,dt,25,[],7, [0.0])

    pred, h = model(initdir, input)
    loss = criterion(pred, output)
    print(f"Loss on test data: {loss.item():.4f}")
    hidden_states = np.array(h.cpu().detach().numpy())
    import pdb; pdb.set_trace()

    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    np_output = np.transpose(output.detach().cpu().numpy(), (2, 1, 0))
    np_output = np.reshape(np_output, (pred.shape[0], -1))
    radsActual = np.unwrap(np.arctan2(np_output[0], np_output[1]))

    plt.plot(radsActual, label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Prediction Visualization')
    plt.legend()
    plt.savefig('performance_fakeAngs2.png')

    plotActivations(hidden_states, input, output)

    del initdir 
    del input 
    del output 

def plotActivations(hidden_states, input, output):
    from scipy.stats import binned_statistic, binned_statistic_2d
    seqlen = hidden_states.shape[0] * hidden_states.shape[1]
    velocities = input.cpu().detach().numpy().reshape(seqlen, -1)[:, 0]
    reshaped_output = output.cpu().detach().numpy().reshape(seqlen, -1)
    headdirs = np.unwrap(np.arctan2(reshaped_output[:,0], reshaped_output[:,1])) % (2 * np.pi)

    print("Hidden states shape: ", hidden_states.shape)
    print("Inputs[i].shape: ", velocities.shape)
    print("Angs[i].shape: ", headdirs.shape)

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )
        
        print(f"Doing it for cell={cell}")
        bs = binned_statistic_2d(velocities, headdirs, activations, bins=[30, 30])
        curr_ax.pcolormesh(bs[1], bs[2], bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs2.png")
    plt.clf()

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )

        print(f"Doing it for cell={cell}, no bins")
        bs = binned_statistic(headdirs, activations)
        curr_ax.plot( (bs[1][1:] + bs[1][:-1]) / 2, bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs_headdirs2.png")
    plt.clf()

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )

        print(f"Doing it for cell={cell}")
        bs = binned_statistic(velocities, activations)
        curr_ax.plot( (bs[1][1:] + bs[1][:-1]) / 2, bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs_vels2.png")
    plt.clf()

if __name__ == '__main__':
    main()
