import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN

NUM_EPOCHS = 1000

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the data and preprocess it
    angs = np.load('angs_smooth.npy')
    dataProcessor = DataPreprocessor(angs, sample_length=700)
    h0 = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
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

        pred = model(h0, input)
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
    dataProcessor = DataPreprocessor(angs, sample_length=angs.shape[1])
    h0 = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    # output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)

    pred = model(h0, input)
    pred = np.squeeze(np.transpose(pred.detach().numpy(), (2, 0, 1)))
    radsOut = np.arctan2(pred[0], pred[1])

    plt.plot(angs[1], label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.legend()
    plt.savefig('performance.png')

if __name__ == '__main__':
    main()
