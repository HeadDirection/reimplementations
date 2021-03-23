import numpy as np
import time
import matplotlib.pyplot as plt

from DataPreprocessor import DataPreprocessor

class AVDataGenerator():
    def __init__(self, T=10000, dt=25, mean=0, sigma=0.03, momentum=0.8):
        """
        T - number of timesteps
        dt - time between consecutive timesteps, in milliseconds
        sigma - hyperparameter used in AV update formula
        momentum - hyperparameter used in AV update formula

        AV update formula: AV(t) = sigma * X + momentum * AV(t-1)
        """ 
        self.T = T
        self.dt = dt
        self.mean = mean
        self.sigma = sigma
        self.momentum = momentum
    
    def GenerateAngs(self):
        """
        Returns a (2, T) numpy array
        First row is time in milliseconds
        Second row is head direction in radians
        """
        seed = int(time.time_ns()) % (2**32)
        np.random.seed(seed)

        initTime = 0
        initRad = np.random.uniform(-np.pi, np.pi)
        initVel = np.random.uniform(-0.5, 0.5)

        times = np.array([initTime])
        rads = np.array([initRad])
        vels = np.array([initVel])

        for t in range(self.T - 1):
            times = np.append(times, times[t] + self.dt)
            if (rads[t] > 20 and vels[t] > 0) or (rads[t] < -20 and vels[t] < 0):
                vels[t] = -2 * vels[t]
            rads = np.append(rads, rads[t] + (self.dt * vels[t]))
            vels = np.append(vels, (np.random.normal(self.mean, self.sigma)) + self.momentum * vels[t])

        angs = np.transpose(np.column_stack((times, rads)))
        return angs

def main():
    realAngs = np.load('angs_smooth.npy') - 2 * np.pi
    plt.plot(np.unwrap(realAngs[1]), label='real angs')
    diffs = realAngs[1][1:] - realAngs[1][:-1]
    print("realAngs avg diff: ", np.mean(diffs))
    print("realAngs stddev: ", np.std(diffs))
    print("realAngs range: ", np.min(realAngs[1]), np.max(realAngs[1]))

    for i in range(10):
        datagen = AVDataGenerator(T=realAngs.shape[1], dt=25, mean=np.mean(diffs)/100, sigma=np.std(diffs)/10, momentum=0)
        # if i == 0:
        #     fakeAngs = realAngs
        # else:
        fakeAngs = datagen.GenerateAngs()
        dataProcessor = DataPreprocessor(fakeAngs, sample_length=700, normalize=False)
        fakeOutputs = dataProcessor.GetTrainingOutputs()
        fakeOutputs = np.transpose(fakeOutputs, (2, 1, 0))
        fakeOutputs = np.reshape(fakeOutputs, (fakeOutputs.shape[0], -1))
        fakeOutputs = np.unwrap(np.arctan2(fakeOutputs[0], fakeOutputs[1]))
        plt.plot(fakeOutputs, label='fake angs')
    
    # fakeAngs = datagen.GenerateAngs()
    # plt.plot(fakeAngs[1], label='fake angs 2')

    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('AVDataGenerator Visualization')
    plt.legend(prop={'size': 6})
    plt.savefig('datagen.png')

if __name__ == '__main__':
    main()

