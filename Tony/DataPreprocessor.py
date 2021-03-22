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
        return np.expand_dims(self.batches[1,:,2:], 0)
    
    def GetTrainingInputs(self):
        return self.batches[:,:,1]

    def GetTrainingOutputs(self):
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
        
