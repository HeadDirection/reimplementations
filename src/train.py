import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN

NUM_EPOCHS = 100
USE_FAKE_DATA = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the data 
    if USE_FAKE_DATA:
        datagen = AVDataGenerator(T=40000)
        angs = datagen.GenerateAngs()
    else:
        angs = np.load('angs_smooth.npy')

    # Preprocess the data
    dataProcessor = DataPreprocessor(angs, sample_length=700, normalize=not USE_FAKE_DATA)
    initdir = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)

    # Define the model and optimizer
    model = SingleLayerCTRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Train
    losses = []
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        pred = model(initdir, input)
        loss = criterion(pred, output)

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
    testAngs = np.load('angs_smooth.npy')
    TestCTRNN(testAngs, model, device)

def TestCTRNN(angs, model, device):
    # TODO: Should this always be normalized=True? Or keep consistent with whether training was normalized?
    dataProcessor = DataPreprocessor(angs, sample_length=700, normalize=not USE_FAKE_DATA) 
    initdir = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    # output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)
    
    pred = model(initdir, input)
    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    plt.plot(angs[1], label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Prediction Visualization')
    plt.legend()
    plt.savefig('performance.png')

if __name__ == '__main__':
    main()
