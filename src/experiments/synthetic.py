from src.scm import SCM
from src.Distributions import MixtureOfGaussians, Gaussian
from src.scm import MLP1, SCM_Trainer

from scipy.special import expit

import numpy as np
import os
import torch


class LinearTemporalSCM(SCM):
    """ Simple synthetic SCM implementing a stochastic process. """
    def __init__(self, alpha=0.01, beta_linear=1.0, beta_seasonal=0.0, seed=2024, **kwargs):
        super().__init__(**kwargs)

        # Components needed for the trend
        self.alpha = alpha # Strength of the trend component
        self.beta1 = beta_linear # Strength of the linear trend
        self.beta2 = beta_seasonal # Strength of the seasonal trend

        self.U = [
            MixtureOfGaussians([[-2, 1.5], [1,1]], seed=seed),
            Gaussian(0, 0.1, seed=seed+42),
            Gaussian(0, 1, seed=seed+95)
        ]

        self.f = [
                    lambda X, U1, t: 0.5*X[t-1, :, 0] + U1,
                    lambda X, U2, t: 0.5*X[t-1, :, 1] -0.25 * X[t, :, 0] + U2, # D x N
                    lambda X, U3, t: 0.5*X[t-1, :, 2] + 0.05 * X[t, :, 0] + 0.25 * X[t, :, 1] + U3,
                ]

        self.inv_f = [
                        lambda X, t: X[t, :, 0] - 0.5*X[t-1, :, 0],
                        lambda X, t: X[t, :, 1] - 0.5*X[t-1, :, 1] + 0.25 * X[t, :, 0],
                        lambda X, t: X[t, :, 2] - 0.5*X[t-1, :, 2] - 0.05 * X[t, :, 0] - 0.25 * X[t, :, 1],
                    ]
        
        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]) - self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t))), 
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X - (torch.zeros_like(X[0]) - self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t)))),
        ]

        # TODO: if we allow this to change at each time step, we are introducing distribution shift in the process
        self._mean_y = 1.8679321 #-0.3107163 # mean (X1+X2+X3) taken by sampling 1000 individuals at the 0th timestep

        self.mean = torch.zeros(3)
        self.std = torch.ones(3)

        self.actionable = [0,1,2]
        self.soft_interv = [True, True, True]

    def sample_U(self, N, T, S=1):
        U1 = self.U[0].sample((S,T,N))
        U2 = self.U[1].sample((S,T,N))
        U3 = self.U[2].sample((S,T,N))

        return np.stack([U1, U2, U3], axis=-1) # T x N x D
    
    def get_descendant_mask(self, actionable):
        N, D = actionable.shape
        descendant_mask = torch.ones(N, D)
        for i in range(D):
            if actionable[0, i] == 1:
                if i == 0:
                    descendant_mask[:, [0, 1, 2]] = 0
                elif i == 1:
                    descendant_mask[:, [1, 2]] = 0
                else:
                    descendant_mask[:, [2]] = 0
        return descendant_mask

    def label(self, X):
        _generator = np.random.default_rng(2024)
        probabilities = 1/(1 + np.exp(-2.5 * (X[:, :, 0]+X[:, :, 1]+X[:, :, 2]/self._mean_y)))
        return _generator.binomial(1, probabilities)

class NonLinearTemporalSCM(LinearTemporalSCM):

    def __init__(self, alpha=0.01, beta_linear=1.0, beta_seasonal = 0.0, seed=2024) -> None:
        
        super().__init__(alpha, beta_linear, beta_seasonal, seed=seed)

        self.U = [
            MixtureOfGaussians(
                [[-2, 1.5], [1,1]]
            ),
            Gaussian(0, 0.1),
            Gaussian(0, 1)
        ]

        self.f = [
            lambda X, U1, t: 0.5*X[t-1, :, 0] + U1,
            lambda X, U2, t: 0.5*X[t-1, :, 1] -1 + 3/(1 + torch.exp(-2*X[t, :, 0])) + U2,
            lambda X, U3, t: 0.5*X[t-1, :, 2] - 0.05 * X[t, :, 0] + 0.25 * torch.pow(X[t, :, 1], 2) + U3
        ]

        self.inv_f = [
            lambda X, t: X[t, :, 0] - 0.5*X[t-1, :, 0],
            lambda X, t: X[t, :, 1] - 0.5*X[t-1, :, 1] + 1 - 3/(1+ torch.exp(-2*X[t, :, 0])),
            lambda X, t: X[t, :, 2] - 0.5*X[t-1, :, 2] + 0.05 * X[t, :, 0] - 0.25 * torch.pow(X[t, :, 1], 2)
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]) - self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t))), 
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X - (torch.zeros_like(X[0]) - self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t)))),
        ]

        # TODO: if we allow this to change at each time step, we are introducing distribution shift in the process
        self._mean_y = 3.4491508 # mean (X1+X2+X3) taken by sampling 1000 individuals at the 0th timestep

class LinearTemporalSCMVariance(LinearTemporalSCM):

    def __init__(self, variance=1, alpha=0.0, beta_linear=0.0, beta_seasonal=0, **kwargs):
        super().__init__(alpha, beta_linear, beta_seasonal, **kwargs)

        self.variance = variance

        self.U = [
            MixtureOfGaussians([[-2, variance], [1,variance]], weights=np.array([0.5*variance, 1-0.5*variance])),
            Gaussian(0, variance),
            Gaussian(0, variance)
        ]

class NonLinearTemporalSCMVariance(NonLinearTemporalSCM):

    def __init__(self, variance=1, alpha=0.0, beta_linear=0.0, beta_seasonal=0, **kwargs):
        super().__init__(alpha, beta_linear, beta_seasonal, **kwargs)

        self.variance = variance

        self.U = [
            MixtureOfGaussians([[-2, variance*1.5], [1,variance*1]], weights=np.array([0.5*variance, 1-0.5*variance])),
            Gaussian(0, variance*0.1),
            Gaussian(0, variance*1)
        ]

class LearnedTemporalSCM(LinearTemporalSCM):
    """
    SCM for the Adult data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Inputs:
        - linear: whether to fit linear or non-linear structural equations
    """
    def __init__(self, linear=False, seed=2024):

        super().__init__()

        self.linear = linear
        self.f = []
        self.inv_f = []

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]), 
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
        ]

        self.U = [
            Gaussian(0, 1, seed=seed+42),
            Gaussian(0, 1, seed=seed+42),
            Gaussian(0, 1, seed=seed+95)
        ]

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(2, 1), torch.nn.Linear(3, 1), torch.nn.Linear(4, 1)
        return MLP1(2, 5), MLP1(3, 5), MLP1(4, 5)

    def fit_eqs(self, X, output_name="linear", path=".", force_train=False):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X2 = f1(X1, U2, t)
            X3 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        if os.path.isfile(os.path.join(path, f"{output_name}_{model_type}_f1.pth")) and not force_train:
            print('Fitted SCM already exists')
            return

        mask_1 = [0, 6]
        mask_2 = [1, 3, 6] # x_1^t-1, x_0^t, t
        mask_3 = [2, 3, 4, 6]

        f1, f2, f3 = self.get_eqs()

        # Regenerate the data such to be in the correct format to train
        # the structural equations
        rearranged_data = []
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                 if t == 0:
                    rearranged_data.append(
                        np.zeros_like(X[t, i, :]).tolist()+X[t, i, :].tolist()+[t]
                    )
                 else:
                    rearranged_data.append(
                        X[t-1, i, :].tolist()+X[t, i, :].tolist()+[t]
                    )
        X = torch.Tensor(rearranged_data)

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        train_epochs = 10
        trainer = SCM_Trainer(verbose=True, print_freq=1, lr=0.005)
        trainer.train(f1,
                      X[id_train][:, mask_1],
                      X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_1],
                      X[id_test, 3].reshape(-1, 1),
                      train_epochs)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 4].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 4].reshape(-1, 1), train_epochs)
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 5].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 5].reshape(-1, 1), train_epochs)

        if path is not None:

            torch.save(f1.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f1.pth"))
            torch.save(f2.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f2.pth"))
            torch.save(f3.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f3.pth"))

        self.set_eqs(f1, f2, f3) # Build the structural equations

    def load(self, output_name="linear", path="."):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2, f3 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f1.pth"), weights_only=True))
        f2.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f2.pth"), weights_only=True))
        f3.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f3.pth"), weights_only=True))

        self.set_eqs(f1, f2, f3)

    def set_eqs(self, f1, f2, f3):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1, self.f2, self.f3 = f1, f2, f3

        self.f = [lambda X, U1, t: f1(torch.cat([X[t-1, :, [0]], torch.ones((X.shape[1], 1)) * t], 1)).flatten() + U1,
                  lambda X, U2, t: f2(torch.cat([X[t-1, :, [1]], X[t, :, [0]], torch.ones((X.shape[1], 1)) * t], 1)).flatten() + U2,
                  lambda X, U3, t: f3(torch.cat([X[t-1, :, [2]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * t], 1)).flatten() + U3,
                  ]

        self.inv_f = [
                      lambda X, t: X[t, :, 0] - f1(torch.cat([X[t-1, :, [0]], torch.ones((X.shape[1], 1)) * t], 1)).flatten(),
                      lambda X, t: X[t, :, 1] - f2(torch.cat([X[t-1, :, [1]], X[t, :, [0]], torch.ones((X.shape[1], 1)) * t], 1)).flatten(),
                      lambda X, t: X[t, :, 2] - f3(torch.cat([X[t-1, :, [2]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * t], 1)).flatten(),
                      ]