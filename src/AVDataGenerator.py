import numpy as np
import time
import matplotlib.pyplot as plt

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
        initRad = np.random.uniform(-3, 3)
        initVel = np.random.uniform(-0.5, 0.5)

        times = np.array([initTime])
        rads = np.array([initRad])
        vels = np.array([initVel])

        for t in range(self.T - 1):
            times = np.append(times, times[t] + self.dt)
            if (rads[t] > 20 and vels[t] > 0) or (rads[t] < -20 and vels[t] < 0):
                vels[t] = -vels[t]
            rads = np.append(rads, rads[t] + (self.dt * vels[t]))
            vels = np.append(vels, (np.random.normal(self.mean, self.sigma)) + self.momentum * vels[t])

        angs = np.transpose(np.column_stack((times, rads)))
        return angs

def main():
    realAngs = np.load('angs_smooth.npy')
    plt.plot(np.unwrap(realAngs[1]), label='real angs')
    diffs = realAngs[1][1:] - realAngs[1][:-1]
    print("realAngs avg diff: ", np.mean(diffs))
    print("realAngs stddev: ", np.std(diffs))
    print("realAngs range: ", np.min(realAngs[1]), np.max(realAngs[1]))

    datagen = AVDataGenerator(T=realAngs.shape[1], dt=25, mean=np.mean(diffs)/100, sigma=np.std(diffs)/10, momentum=0)
    fakeAngs = datagen.GenerateAngs()
    plt.plot(fakeAngs[1], label='fake angs')

    plt.xlabel('Timestep (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('AVDataGenerator Visualization')
    plt.legend()
    plt.savefig('datagen.png')

if __name__ == '__main__':
    main()

