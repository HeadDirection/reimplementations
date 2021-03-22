import numpy as np

class AVDataGenerator():
    def __init__(self, T=10000, dt=25, sigma=0.03, momentum=0.8):
        """
        T - number of timesteps
        dt - time between consecutive timesteps, in milliseconds
        sigma - hyperparameter used in AV update formula
        momentum - hyperparameter used in AV update formula

        AV update formula: AV(t) = sigma * X + momentum * AV(t-1)
        """ 
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.momentum = momentum
    
    def GenerateAngs(self):
        """
        Returns a (2, T) numpy array
        First row is time in milliseconds
        Second row is head direction in radians
        """
        initTime = 0
        initRad = np.random.uniform(-3, 3)
        initVel = np.random.uniform(-0.5, 0.5)

        times = np.array([initTime])
        rads = np.array([initRad])
        vels = np.array([initVel])

        for t in range(self.T - 1):
            times = np.append(times, times[t] + self.dt)
            rads = np.append(rads, rads[t] + (self.dt * vels[t]))
            vels = np.append(vels, (self.sigma * np.random.normal(0, 1)) + self.momentum * vels[t])

        angs = np.transpose(np.column_stack((times, rads)))
        return angs
