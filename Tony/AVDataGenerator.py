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
    
    def AngsToAv(self, angs, printsamples=False):
        """
        Given a (2, T) angs array, calculate the angular velocity in 
        radians / millisecond using the Euler method
        """
        a2 = angs[:, 1:]
        a1 = angs[:, :-1]
        diffs = (a2 - a1) 
        radsPerMs = diffs[1] / diffs[0]
        if printsamples:
            print("samples: time in ms ", angs[0][0], angs[0][1], angs[0][2])
            print("samples: ang in rad ", angs[1][0], angs[1][1], angs[1][2])
            print("Quick Test to Confirm that First Element of tradsPerMs Array is Correct: ")
            print("\trad per ms by hand = ", (angs[1][1]-angs[1][0])/(angs[0][1]-angs[0][0]))
            print("\tradsPerMs[0] =", radsPerMs[0])
            print("\tradsPerMs shape ", radsPerMs.shape)
        return radsPerMs

    def AngsToInitDir(self, angs):
        """
        Given a (2, T) angs array, returns [sin(theta), cos(theta)] where theta
        is the initial angle.
        """
        return np.array([np.sin(angs[1][0]), np.cos(angs[1][0])])


def main():
    datagen = AVDataGenerator()
    datagen.AngsToAv(datagen.GenerateAngs(), printsamples=True)

if __name__ == '__main__':
    main()
