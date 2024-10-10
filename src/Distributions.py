import numpy as np
import scipy.stats as ss

class Gaussian:

    def __init__(self, mean=0, var=1, seed=2024):
        self._mean = mean
        self._var = var
        self._generator = np.random.default_rng(seed)
    
    def sample(self, shape):
        return self._generator.normal(
            self._mean, self._var, size=shape
        )

class Binomial:

    def __init__(self, n=1, p=0.5, seed=2024):
        self._n = n
        self._p = p
        self._generator = np.random.default_rng(seed)
    
    def sample(self, shape):
        return self._generator.binomial(
            self._n, self._p, size=shape
        )

class Poisson:

    def __init__(self, lam=1,seed=2024):
        self._lambda = lam
        self._generator = np.random.default_rng(seed)
    
    def sample(self, shape):
        return self._generator.poisson(
            self._lambda, size=shape
        )

class Gamma:

    def __init__(self, shape=1, scale=1.0, seed=2024):
        self._shape = shape
        self._scale = scale
        self._generator = np.random.default_rng(seed)
    
    def sample(self, shape):
        return self._generator.gamma(
            self._shape, self._scale, size=shape
        )

class MixtureOfGaussians:

    def __init__(self, components = [[0,1], [0,1]], weights=None, seed=2024) -> None:
        
        self._components = components
        self._weights = np.ones(len(components), dtype=np.float64)/len(components) if weights is None else weights
        
        # Specify seed for the random elements
        self._norms = [Gaussian(*(self._components[i]), seed=seed+i) for i in range(len(components))]        
        self._generator = np.random.default_rng(seed)
    
    def sample(self, shape):

        """The merging script was obtained by querying ChatGPT with the following prompt
        
        Consider M arrays of size (S,T,N,D), and an additional array k of size (S,T,N,D).
        The array k contains only values between [0, M-1]. 
        Can you write a function which merges the M array by picking the values corresponding to the array k?
        Write the function using python and numpy
        
        """
        
        S, T, N = shape
        
        # Sample indices according to the mixture weights
        mixture_idx = self._generator.choice(len(self._weights), size=(S, T, N), replace=True, p=self._weights)

        # Generate samples for each Gaussian component
        U = np.array([norm.sample(shape) for norm in self._norms])  # Shape: (S, T, N)

        # Efficiently index the samples using the mixture indices
        merged = U[mixture_idx, np.arange(S)[:, None, None], np.arange(T)[None, :, None], np.arange(N)[None, None, :]]

        return merged