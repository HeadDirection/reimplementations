from AVDataGenerator import AVDataGenerator
from ContinuousTimeRNN import ContinuousTimeRNN

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

NUM_ITERS = 50

def main():
    model = ContinuousTimeRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    datagen = AVDataGenerator(T=5000)
    losses = []
    for i in range(NUM_ITERS):
        print(f"Training iteration {i}")

        angs = datagen.GenerateAngs()
        av = datagen.AngsToAv(angs)
        initdir = datagen.AngsToInitDir(angs)
        expected = np.expand_dims(np.expand_dims(angs[1][:-1], 1), 1)
        velocities = np.expand_dims(np.expand_dims(av, 1), 1)
        initdir = np.expand_dims(initdir, 0)

        output = model(torch.from_numpy(initdir).float(), torch.from_numpy(velocities).float())
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

    testAngs = np.load('angs_smooth.npy')
    testAv = datagen.AngsToAv(testAngs)
    testInitdir = datagen.AngsToInitDir(testAngs)
    testVelocities = np.expand_dims(np.expand_dims(testAv, 1), 1)
    output = model(torch.from_numpy(testInitdir).float(), torch.from_numpy(testVelocities).float())

    plt.plot(testAngs[1], label='ground truth')
    plt.plot(np.squeeze(output.detach().numpy()), label='predicted')
    plt.legend()
    plt.savefig('performance.png')

if __name__ == '__main__':
    main()
