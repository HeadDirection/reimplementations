import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN
from simple_data_gen import *

NUM_EPOCHS = 300

def main():
    print("fakeAngs2 v7")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the data 
    train_data_size = 66
    train_data_len = 700
    dt = 0.0025
    initdir, input, output = gen_batch(train_data_size,train_data_len,dt,25,[],7, [0.0])

    # Define the model and optimizer
    model = SingleLayerCTRNN(store_h=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Train
    losses = []
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        pred, h = model(initdir, input)
        loss = criterion(pred, output)

        print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}' )
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    
    # Save the model
    torch.save(model.state_dict(), 'activations_fakedata_0025.pt')
    # model.load_state_dict(torch.load('activations_fakedata'))

    # Graph the losses
    print(f"Losses: {losses}")
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('loss_ctrnn_0025.png')
    plt.clf()

    # TestCTRNN(model, criterion)
    
    del initdir
    del input
    del output

def TestCTRNN(model, criterion):
    train_data_size = 1400
    train_data_len = 700
    dt = 0.0025
    initdir, input, output = gen_batch(train_data_size,train_data_len,dt,25,[],7, [0.0])

    pred, h = model(initdir, input)
    loss = criterion(pred, output)
    print(f"Loss on test data: {loss.item():.4f}")
    hidden_states = np.array(h.cpu().detach().numpy())

    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    np_output = output.detach().cpu().numpy()
    radsActual = np.unwrap(np.arctan2(np_output[:,:,0].flatten(), np_output[:,:,1].flatten()))

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
