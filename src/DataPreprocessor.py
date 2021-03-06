import numpy as np

class DataPreprocessor():
    def __init__(self, angs, sample_length=700, normalize=False):
        """
        angs (2, T+1): array of two row vectors, where first row is time in ms 
                        and second row is angle in radians
        sample_length:  will split angs into chunks of this size for training
        normalize: if True, will normalize angular velocities
        """
        self.timesteps = angs[0]
        self.dt = np.mean(np.diff(angs[0]))

        self.angs = angs 
        self.av = self.AngsToAv(angs)
        self.sines, self.cosines = self.AngsToSinCos(angs)
        if normalize:
            self.av = self.av / np.mean(np.abs(self.av))

        self.sample_length = sample_length
        self.num_samples = angs.shape[1] // self.sample_length
        self.T = self.num_samples * self.sample_length

        combined = np.vstack([self.timesteps[:self.T], self.av[:self.T], self.sines[:self.T], self.cosines[:self.T]])
        self.batches = np.reshape(combined, (combined.shape[0], self.num_samples, self.sample_length))
        self.batches = np.transpose(self.batches, (2,1,0))

    def GetInitialInput(self):
        """
        Returns representation of initial input (e.g. sin and cos of initial angle)
        Shape: (1, N, dim(initdir)), 
            where dim(initdir)=2 if it's just sin and cos, and
            where N is the number of samples
        """
        return np.expand_dims(self.batches[1,:,2:], 0)
    
    def GetTrainingInputs(self):
        """
        Returns inputs (e.g. velocities)
        Shape: (T, N, dim(input)), where dim(input)=1 if input is just velocity
        """
        # vels = np.expand_dims(self.batches[:,:,1], 2)
        # angsraw = self.angs[1][:self.T-1]
        # angsraw = np.insert(angsraw, 0, angsraw[0])
        # angsraw = angsraw.reshape(self.sample_length, -1)
        # angs = np.expand_dims(((-angsraw) % (2 * np.pi)) - np.pi, 2)
        # return np.concatenate((vels, angs), axis=2)
        return np.expand_dims(self.batches[:,:,1], 2)

    def GetTrainingOutputs(self):
        """
        Returns expected outputs (e.g. sin and cos of the correct angles)
        Shape: (T, N, dim(output)), 
            where dim(output)=2 if output is sin and cos of correct angle
        """
        return self.batches[:,:,2:]

    def AngsToAv(self, angs):
        """
        Given a (2, T+1) angs array, calculate the angular velocity in 
        radians / millisecond using the Euler method. Returns (T,) vector 
        """
        a2 = angs[:, 1:]
        a1 = angs[:, :-1]
        diffs = (a2 - a1) 
        radsPerMs = diffs[1] / diffs[0]
        return radsPerMs
    
    def AngsToSinCos(self, angs):
        """
        Given a (2, T+1) angs array, return the sine and cosine of the radians
        row. Returns two (T,) vectors.
        """
        return np.ndarray.flatten(np.sin(angs[1:])), np.ndarray.flatten(np.cos(angs[1:]))
        
