import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from AVDataGenerator import AVDataGenerator
from DataPreprocessor import DataPreprocessor
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN

NUM_EPOCHS = 1
TRAINING_BATCHES = 2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Processing training data...")
    angs = np.load('angs_smooth.npy')
    datagen = AVDataGenerator(T=angs.shape[1])
    
    initdirs = []
    inputs = []
    outputs = []
    
    for i in range(TRAINING_BATCHES):
        dataProcessor = DataPreprocessor(datagen.GenerateAngs(), sample_length=700, normalize=False)
        initdirs.append(torch.from_numpy(dataProcessor.GetInitialInput()).float())
        inputs.append(torch.from_numpy(dataProcessor.GetTrainingInputs()).float())
        outputs.append(torch.from_numpy(dataProcessor.GetTrainingOutputs()).float())
        print("Outputs: ", outputs[i][0][0])

    initdirs = torch.stack(initdirs).to(device)
    inputs = torch.stack(inputs).to(device)
    outputs = torch.stack(outputs).to(device)

    print("Defining the model...")
    model = SingleLayerCTRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-6)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print("Training...")
    losses = []
    for epoch in range(NUM_EPOCHS):
        for batch in range(TRAINING_BATCHES):
            optimizer.zero_grad()

            pred = model(initdirs[batch], inputs[batch])
            loss = criterion(pred, outputs[batch])

            print (f'Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch+1}/{TRAINING_BATCHES}] Loss: {loss.item():.4f}' )
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
    
    print("Graphing the losses to loss_ctrnn.py...")
    print(f"Losses: {losses}")
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('loss_ctrnn.png')
    plt.clf()

    print("Testing the model...")
    testAngs = np.load('angs_smooth.npy')
    TestCTRNN(testAngs, model, device, outputs)

def TestCTRNN(angs, model, device, training_outputs):
    dataProcessor = DataPreprocessor(angs, sample_length=700, normalize=True) 
    initdir = torch.from_numpy(dataProcessor.GetInitialInput()).float().to(device)
    input = torch.from_numpy(dataProcessor.GetTrainingInputs()).float().to(device)
    # output = torch.from_numpy(dataProcessor.GetTrainingOutputs()).float().to(device)
    
    pred = model(initdir, input)
    pred = np.transpose(pred.detach().cpu().numpy(), (2, 1, 0))
    pred = np.reshape(pred, (pred.shape[0], -1))
    radsOut = np.unwrap(np.arctan2(pred[0], pred[1]))

    print("Graphing test performance to performance.png...")
    plt.plot(angs[1], label='ground truth', color='blue')
    plt.plot(radsOut, label='predicted', color='orange')
    for i, output in enumerate(training_outputs):
        out = np.transpose(output.detach().cpu().numpy(), (2, 1, 0))
        out = np.reshape(pred, (pred.shape[0], -1))
        plt.plot(np.unwrap(np.arctan2(out[0], out[1])) + (i+2), label=f'training data {i}')
    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Prediction Visualization')
    plt.legend()
    plt.savefig('performance.png')

if __name__ == '__main__':
    main()
