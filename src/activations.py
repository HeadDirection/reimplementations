import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN
from simple_data_gen import *

angs = []
initdirs = []
inputs = []
outputs = []

test_angs = []
test_initdirs = []
test_inputs = []
test_outputs = []

def GenerateFakeAngs():
    realAngs = np.load('angs_smooth.npy') - 2 * np.pi
    diffs = realAngs[1][1:] - realAngs[1][:-1]

    sigmas = [np.std(diffs)/30, np.std(diffs)/10, np.std(diffs)/6, np.std(diffs)/4, np.std(diffs)/3, np.std(diffs)/2, np.std(diffs)]

    for sigma in sigmas:
        timesteps = 100000
        datagen = AVDataGenerator(T=timesteps, dt=25, mean=np.mean(diffs)/100, sigma=sigma, momentum=0)
        generatedAngs = datagen.GenerateAngs()
        plt.plot(generatedAngs[1])
        print("Generated angs shape: ", generatedAngs.shape)
        trainsize = int(0.8 * timesteps)
        dataProcessor = DataPreprocessor(generatedAngs[:,:trainsize], sample_length=700, normalize=True)
        # torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
        angs.append(generatedAngs[1,:trainsize])
        initdirs.append(dataProcessor.GetInitialInput())
        inputs.append(dataProcessor.GetTrainingInputs())
        outputs.append(dataProcessor.GetTrainingOutputs())

        testDataProcessor = DataPreprocessor(generatedAngs[:,trainsize:], sample_length=700, normalize=True)
        # torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
        test_angs.append(generatedAngs[1,trainsize:])
        test_initdirs.append(testDataProcessor.GetInitialInput())
        test_inputs.append(testDataProcessor.GetTrainingInputs())
        test_outputs.append(testDataProcessor.GetTrainingOutputs())
    
    plt.savefig('fakeangs.png')
    plt.clf()

def TestCTRNN(idx, model, criterion, device):
    initdir = torch.from_numpy(test_initdirs[idx]).float().to(device)
    input = torch.from_numpy(test_inputs[idx]).float().to(device)
    output = torch.from_numpy(test_outputs[idx]).float().to(device)
    
    pred, _ = model(initdir, input)
    loss = criterion(pred, output)
    print(f"EDITED Loss on real data: {loss.item():.4f}")
    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    plt.plot(test_angs[0][:radsOut.shape[0]], label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Prediction Visualization')
    plt.legend()
    plt.savefig(f'performance_test{idx}.png')

    del initdir 
    del input
    del output
    return radsOut

def Train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 70

    for i in range(len(inputs)):
        model = SingleLayerCTRNN(store_h=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
        criterion = nn.MSELoss()
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()

        initdir = torch.from_numpy(initdirs[i]).float().to(device)
        input = torch.from_numpy(inputs[i]).float().to(device)
        output = torch.from_numpy(outputs[i]).float().to(device)

        losses = []
        hidden_states = np.array([], dtype=float)
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()

            pred, h = model(initdir, input)
            loss = criterion(pred, output)
            hidden_states = np.array(h.cpu().detach().numpy())

            print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}' )
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        print(f"Losses: {losses}")
        _ = TestCTRNN(i, model, criterion, device)

        from scipy.stats import binned_statistic_2d
        seqlen = hidden_states.shape[0] * hidden_states.shape[1]
        velocities = inputs[i].reshape(seqlen, -1)[:, 0]
        headdirs = angs[i][:seqlen] % (2 * np.pi)
        print("Hidden states shape: ", hidden_states.shape)
        print("Inputs[i].shape: ", inputs[i].shape)
        print("Angs[i].shape: ", angs[i].shape)

        _, ax = plt.subplots(10, 10, figsize=(20,15))
        for cell in range(100):
            x,y = np.divmod(cell,10)
            curr_ax = ax[x, y]

            activations = torch.relu( torch.tanh( torch.from_numpy( hidden_states.reshape(seqlen, -1)[:, cell] ) ) )

            # import pdb; pdb.set_trace()
            print(f"Doing it for i={i}, cell={cell}")
            bs = binned_statistic_2d(headdirs, velocities, activations, bins=[15, 15])
            curr_ax.pcolormesh(bs[1], bs[2], bs[0])
            curr_ax.set_yticks([])
            curr_ax.set_xticks([])
        
        plt.savefig(f"activations{i}.png")
        plt.clf()
        
        del initdir
        del input
        del output

def main():
    GenerateFakeAngs()
    Train()

if __name__ == '__main__':
    main()


