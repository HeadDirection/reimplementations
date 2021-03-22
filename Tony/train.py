from AVDataGenerator import AVDataGenerator
from ContinuousTimeRNN import ContinuousTimeRNN
from SingleLayerCTRNN import SingleLayerCTRNN

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

NUM_ITERS = 50
NUM_EPOCHS = 50
DIN = 1
DOUT = 2

def main():
    testAngs = np.load('angs_smooth.npy')
    # model = ContinuousTimeRNN()
    model = SingleLayerCTRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    chunksize = testAngs.shape[1] // NUM_ITERS
    datagen = AVDataGenerator(T=5000)
    losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")
        initdirs = np.zeros((1, NUM_ITERS, 2))
        velocities = np.zeros((chunksize-1, NUM_ITERS, DIN))
        expected = np.zeros((chunksize-1, NUM_ITERS, DOUT))

        for i in range(NUM_ITERS):
            rangeStart = i * chunksize
            rangeEnd = rangeStart + chunksize
            angs = testAngs[:, rangeStart:rangeEnd]

            # import pdb; pdb.set_trace()
            # angs = datagen.GenerateAngs()
            initdir = datagen.AngsToInitDir(angs)
            initdirs[:, i] = initdir

            av = datagen.AngsToAv(angs)
            velocities[:, i, 0] = av

            sines = np.sin(angs[1][:-1])
            cosines = np.cos(angs[1][:-1])
            expected[:, i, 0] = sines
            expected[:, i, 1] = cosines

        optimizer.zero_grad()

        output = model(torch.from_numpy(initdirs).float(), torch.from_numpy(velocities).float())
        loss = criterion(output, torch.from_numpy(expected).float())

        print(f"\tLoss for iteration {i} is {loss.item()}")
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    
    print(f"Losses: {losses}")
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('loss_ctrnn.png')
    plt.clf()

    # TODO change
    testAv = datagen.AngsToAv(testAngs)
    testInitdir = datagen.AngsToInitDir(testAngs)
    testVelocities = np.expand_dims(np.expand_dims(testAv, 1), 1)
    output = model(torch.from_numpy(testInitdir).float(), torch.from_numpy(testVelocities).float())
    output = np.squeeze(np.transpose(output.detach().numpy(), (2, 0, 1)))
    radsOut = np.arctan2(output[0], output[1])

    plt.plot(testAngs[1], label='ground truth')
    plt.plot(radsOut, label='predicted')
    plt.legend()
    plt.savefig('performance.png')

if __name__ == '__main__':
    main()
