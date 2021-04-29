import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN

NUM_EPOCHS = 300

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the data 
    realAngs = np.load('angs_smooth.npy')
    realAngs[1] -= 2 * np.pi
    diffs = realAngs[1][1:] - realAngs[1][:-1]
    datameanmean = np.mean(diffs)/100
    datameansigma = np.std(diffs)/10
    datagen = AVDataGenerator(T=realAngs.shape[1], dt=25, mean=datameanmean, sigma=datameansigma, momentum=0)

    # Preprocess the data
    fakeAngs = datagen.GenerateAngs()
    plt.plot(fakeAngs[1])
    plt.savefig("fakeAngs.png")
    plt.clf()
    dataProcessor = DataPreprocessor(fakeAngs, sample_length=700, normalize=True)
    initdir = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)

    # Define the model and optimizer
    model = SingleLayerCTRNN(store_h=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Train
    losses = []
    hidden_states = None
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        pred, h = model(initdir, input)
        loss = criterion(pred, output)
        hidden_states = np.array(h.cpu().detach().numpy())

        print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}' )
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    
    # Graph the losses
    print(f"Losses: {losses}")
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('loss_ctrnn.png')
    plt.clf()

    # Test
    testAngs = datagen.GenerateAngs()
    TestCTRNN(testAngs, model, criterion, device)

    from scipy.stats import binned_statistic, binned_statistic_2d
    seqlen = hidden_states.shape[0] * hidden_states.shape[1]
    velocities = input.cpu().detach().numpy().reshape(seqlen, -1)[:, 0]
    headdirs = testAngs[1][:seqlen] % (2 * np.pi)
    print("Hidden states shape: ", hidden_states.shape)
    print("Inputs[i].shape: ", velocities.shape)
    print("Angs[i].shape: ", headdirs.shape)

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )
        # activations = hidden_states.reshape(seqlen, -1)[:, cell]
        # print("No nonlin")
        
        print(f"Doing it for cell={cell}")
        bs = binned_statistic_2d(headdirs, velocities, activations)
        curr_ax.pcolormesh(bs[1], bs[2], bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs.png")
    plt.clf()

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )
        # activations = hidden_states.reshape(seqlen, -1)[:, cell]
        
        # import pdb; pdb.set_trace()
        print(f"Doing it for cell={cell}, no bins")
        bs = binned_statistic(headdirs, activations, bins=[30, 30])
        curr_ax.plot( (bs[1][1:] + bs[1][:-1]) / 2, bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs_headdirs.png")
    plt.clf()

    _, ax = plt.subplots(10, 10, figsize=(20,15))
    for cell in range(100):
        x,y = np.divmod(cell,10)
        curr_ax = ax[x, y]

        activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )
        # activations = hidden_states.reshape(seqlen, -1)[:, cell]
        # print("No nonlin")
        
        # import pdb; pdb.set_trace()
        print(f"Doing it for cell={cell}")
        bs = binned_statistic(velocities, activations)
        curr_ax.plot( (bs[1][1:] + bs[1][:-1]) / 2, bs[0])
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
    
    plt.savefig(f"activations_fakeAngs_vels.png")
    plt.clf()
    
    del initdir
    del input
    del output

def TestCTRNN(angs, model, criterion, device):
    dataProcessor = DataPreprocessor(angs, sample_length=700, normalize=True) 
    initdir = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)
    
    pred, h = model(initdir, input)
    loss = criterion(pred, output)
    print(f"Loss on real data: {loss.item():.4f}")
    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    plt.plot(angs[1], label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Prediction Visualization')
    plt.legend()
    plt.savefig('performance_fakeAngs.png')

    del initdir 
    del input 
    del output 

if __name__ == '__main__':
    main()
