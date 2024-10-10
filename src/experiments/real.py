from src.scm import SCM, MLP1, SCM_Trainer
from src.models.CVAE import CVAE, CVAETrainer
from src.Distributions import Gaussian, Binomial, Poisson, Gamma

import numpy as np
import os
import torch


class SemiSyntheticAdultTemporal(SCM):
    """
    SCM for the Adult data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Inputs:
        - linear: whether to fit linear or non-linear structural equations
    """
    def __init__(self, alpha=0.7, linear_trend=1.0, seasonal_trend=0, linear=False, seed=2024):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(6)
        self.std = torch.ones(6)

        self.actionable = [4, 5]

        # Components needed for the trend
        self.alpha = alpha # Strength of the trend component
        self.beta1 = linear_trend # Strength of the linear trend
        self.beta2 = seasonal_trend # Strength of the seasonal trend

        self.U = [
            Binomial(1, 0.9, seed=seed),
            Gaussian(0, 1, seed=seed+42),
            Binomial(1, 0.9, seed=seed+95),
            Gaussian(0, 0, seed=seed+11),
            Gaussian(0, 1, seed=seed+9),
            Gaussian(0, 1, seed=seed+1),
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]) - self.alpha * (self.beta1 * min(0.01*t, 10) + self.beta2 * np.abs(np.sin(0.5*t))), 
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X - (torch.zeros_like(X[0]) - self.alpha * (self.beta1 * min(0.01*t, 10) + self.beta2 * np.abs(np.sin(0.5*t)))),
        ]

        # See for random generation
        self._generator = np.random.default_rng(seed)

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(3, 1), torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)
        return MLP1(3), MLP1(4), MLP1(4)
    
    def sample_U(self, N, T, S=1):
        U1 = np.tile(self.U[0].sample((S,1,N)), (1, T, 1))
        U2 = np.tile(self.U[1].sample((S,1,N)), (1, T, 1))
        U3 = np.tile(self.U[2].sample((S,1,N)), (1, T, 1))
        U4 = self.U[3].sample((S,T,N))
        U5 = self.U[4].sample((S,T,N))
        U6 = self.U[5].sample((S,T,N))

        return np.stack([U1, U2, U3, U4, U5, U6], axis=-1) # T x N x D
    
    def label(self, X):
        return torch.ones((X.shape[0], X.shape[1], 1))
    
    def get_descendant_mask(self, actionable):
        N, D = actionable.shape
        descendant_mask = torch.ones(N, D)
        for i in range(D):
            if actionable[0, i] == 1:
                if i == 0:
                    descendant_mask[:, [0, 3, 4, 5]] = 0
                elif i == 1:
                    descendant_mask[:, [1, 3, 4, 5]] = 0
                elif i == 2:
                    descendant_mask[:, [2, 4, 5]] = 0
                elif i == 3:
                    descendant_mask[:, [3]] = 0
                elif i == 4:
                    descendant_mask[:, [4]] = 0
                elif i == 5:
                    descendant_mask[:, [5]] = 0
        return descendant_mask
    

    def fit_eqs(self, X, save=None):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        if os.path.isfile(save+model_type+'_f1.pth'):
            print('Fitted SCM already exists')
            return


        mask_1 = [0, 1, 2]
        mask_2 = [0, 1, 2, 3]
        mask_3 = [0, 1, 2, 3]

        f1, f2, f3 = self.get_eqs()

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = self._generator.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        train_epochs = 10
        trainer = SCM_Trainer(verbose=False, print_freq=1, lr=0.005)        
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 3].reshape(-1, 1), train_epochs)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 4].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 4].reshape(-1, 1), train_epochs)
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 5].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 5].reshape(-1, 1), train_epochs)

        if save is not None:

            torch.save(f1.state_dict(), save+model_type+'_f1.pth')
            torch.save(f2.state_dict(), save+model_type+'_f2.pth')
            torch.save(f3.state_dict(), save+model_type+'_f3.pth')

        self.set_eqs(f1, f2, f3) # Build the structural equations

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2, f3 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth', weights_only=True))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth', weights_only=True))
        f3.load_state_dict(torch.load(name + model_type + '_f3.pth', weights_only=True))

        self.set_eqs(f1, f2, f3)

    def set_eqs(self, f1, f2, f3):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1, self.f2, self.f3 = f1, f2, f3

        self.f = [lambda X, U1, t: X[t-1, :, 0] + (t==1)*U1, # This weird step is needed since we assume these values to be fixed
                  lambda X, U2, t: X[t-1, :, 1] + (t==1)*U2,
                  lambda X, U3, t: X[t-1, :, 2] + (t==1)*U3,
                  lambda X, U4, t: (torch.sigmoid(f1(X[t, :, [0,1,2]]).flatten()) > 0.5).float(),
                  lambda X, U5, t: 0.5*X[t-1, :, 4] + f2(X[t, :, [0,1,2,3]]).flatten() + U5,
                  lambda X, U6, t: 0.5*X[t-1, :, 5] + f3(X[t, :, [0,1,2,3]]).flatten() + U6,
                  ]

        self.inv_f = [lambda X, t: X[t, :, 0],
                      lambda X, t: X[t, :, 1],
                      lambda X, t: X[t, :, 2],
                      lambda X, t: X[t, :, 3] - (torch.sigmoid(f1(X[t, :, [0,1,2]]).flatten()) > 0.5).float(),
                      lambda X, t: X[t, :, 4] - 0.5*X[t-1, :, 4] - f2(X[t, :, [0,1,2,3]]).flatten(),
                      lambda X, t: X[t, :, 5] - 0.5*X[t-1, :, 5] - f3(X[t, :, [0,1,2,3]]).flatten(),
                      ]


class LearnedAdultTemporal(SemiSyntheticAdultTemporal):
    
    def __init__(self, linear=False, time_preprocess="linear", seed=2024):
        super().__init__(0, 0, 0, linear, seed)

        self._generator = np.random.default_rng(2024)
        self._time_preprocess = time_preprocess

        self.U = [
            Gaussian(0, 0, seed=seed+11), # We assume these are fixed, taken from the data so no randomness
            Gaussian(0, 0, seed=seed+11),
            Gaussian(0, 0, seed=seed+11),
            Gaussian(0, 0, seed=seed+11),
            Gaussian(0, 1, seed=seed+9),
            Gaussian(0, 1, seed=seed+1),
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X
        ]
    
    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(5, 1), torch.nn.Linear(6, 1), torch.nn.Linear(6, 1)
        return MLP1(5, 10, sigmoid=True), MLP1(6, 10), MLP1(6, 10)
    
    def process_time(self, t):
        if self._time_preprocess == "linear":
            return t
        elif self._time_preprocess == "sin":
            return np.sin(t)
        return t 

    def fit_eqs(self, X, output_name="linear", path=".", force_train=False):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = 'lin' if self.linear else 'mlp'
        if os.path.isfile(os.path.join(path, f"{output_name}_{model_type}_f1.pth")) and not force_train:
            print('Fitted SCM already exists')
            return
        
        print("Fitting SCM equations")

        # [0,1,2,3,4,5, 6,7,8,9,10,11 ,12]
        mask_1 = [3, 6, 7, 8, 12]
        mask_2 = [4, 6, 7, 8, 9, 12] # x_1^t-1, x_0^t, t
        mask_3 = [5, 6, 7, 8, 9, 12]

        f1, f2, f3 = self.get_eqs()

        # Regenerate the data such to be in the correct format to train
        # the structural equations
        rearranged_data = []
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                 if t == 0:
                    rearranged_data.append(
                        np.zeros_like(X[t, i, :]).tolist()+X[t, i, :].tolist()+[self.process_time(t)]
                    )
                 else:
                    rearranged_data.append(
                        X[t-1, i, :].tolist()+X[t, i, :].tolist()+[self.process_time(t)]
                    )
        X = torch.Tensor(rearranged_data)

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = self._generator.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]
        id_train = indices

        train_epochs = 10
        trainer = SCM_Trainer(verbose=True, print_freq=1, lr=0.005, batch_size=N_train//100)        
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 9].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 9].reshape(-1, 1), train_epochs)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 10].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 10].reshape(-1, 1), train_epochs)
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 11].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 11].reshape(-1, 1), train_epochs)

        # Save the structural equations
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

        model_type = 'lin' if self.linear else 'mlp'
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

        self.f = [lambda X, U1, t: X[t-1, :, 0] + (t==1)*X[t, :, 0],
                  lambda X, U2, t: X[t-1, :, 1] + (t==1)*X[t, :, 1],
                  lambda X, U3, t: X[t-1, :, 2] + (t==1)*X[t, :, 2],
                  lambda X, U4, t: f1(torch.cat([X[t-1, :, [3]], X[t, :, [0,1,2]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                  lambda X, U5, t: f2(torch.cat([X[t-1, :, [4]], X[t, :, [0,1,2,3]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten() + U5,
                  lambda X, U6, t: f3(torch.cat([X[t-1, :, [5]], X[t, :, [0,1,2,3]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten() + U6,
                  ]

        self.inv_f = [lambda X, t: X[t, :, 0],
                      lambda X, t: X[t, :, 1],
                      lambda X, t: X[t, :, 2],
                      lambda X, t: X[t, :, 3] - f1(torch.cat([X[t-1, :, [3]], X[t, :, [0,1,2]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 4] - f2(torch.cat([X[t-1, :, [4]], X[t, :, [0,1,2,3]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 5] - f3(torch.cat([X[t-1, :, [5]], X[t, :, [0,1,2,3]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      ]
    

class SemiSyntheticCOMPASTemporal(SCM):
    """
    SCM for the COMPAS data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Age, Gender -> Race, Priors
    Race -> Priors
    Feature names: ['age', 'isMale', 'isCaucasian', 'priors_count']
    """
    def __init__(self, alpha=0.7, linear_trend=1.0, seasonal_trend=0, linear=False, seed=2024):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(4)
        self.std = torch.ones(4)

        self.actionable = [3]

        # Components needed for the trend
        self.alpha = alpha # Strength of the trend component
        self.beta1 = linear_trend # Strength of the linear trend
        self.beta2 = seasonal_trend # Strength of the seasonal trend

        self.U = [
            Poisson(1, seed=seed),
            Binomial(1, 0.8, seed=seed+42),
            Gaussian(0, 1, seed=seed+95),
            Gaussian(0, 1, seed=seed+11),
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]) + self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t))), 
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X - (torch.zeros_like(X[0]) + self.alpha * (self.beta1 * min(0.05*t, 10) + self.beta2 * np.abs(np.sin(0.5*t)))),
        ]

        # See for random generation
        self._generator = np.random.default_rng(seed)

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(2, 1), torch.nn.Linear(3, 1)
        return MLP1(2), MLP1(3)
    
    def get_descendant_mask(self, actionable):
        N, D = actionable.shape
        descendant_mask = torch.ones(N, D)
        for i in range(D):
            if actionable[0, i] == 1:
                if i == 0:
                    descendant_mask[:, [0, 2, 3]] = 0
                elif i == 1:
                    descendant_mask[:, [1, 2, 3]] = 0
                elif i == 2:
                    descendant_mask[:, [2, 3]] = 0
                elif i == 2:
                    descendant_mask[:, [3]] = 0
        return descendant_mask
    
    def sample_U(self, N, T, S=1):
        U1 = np.tile(self.U[0].sample((S,1,N)), (1, T, 1))
        U2 = np.tile(self.U[1].sample((S,1,N)), (1, T, 1))
        U3 = np.tile(self.U[2].sample((S,1,N)), (1, T, 1))
        U4 = self.U[3].sample((S,T,N))

        return np.stack([U1, U2, U3, U4], axis=-1) # T x N x D
        
    def label(self, X):
        return torch.ones((X.shape[0], X.shape[1], 1))

    def fit_eqs(self, X, save=None):
        raise NotImplementedError

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth', weights_only=True))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth', weights_only=True))

        self.set_eqs(f1, f2)

    def set_eqs(self, f1, f2):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1 = f1
        self.f2 = f2

        self.f = [lambda X, U1, t: X[t-1, :, 0] + (t==1)*U1,
                  lambda X, U2, t: X[t-1, :, 1] + (t==1)*U2,
                  lambda X, U3, t: X[t-1, :, 2] + (t==1)*(torch.sigmoid(f1(X[t, :, [0,1]]).flatten()) + U3),
                  lambda X, U4, t: 0.5*X[t-1, :, 3] + f2(X[t, :, [0,1,2]]).flatten() + U4,
                  ]

        self.inv_f = [lambda X, t: X[t, :, 0],
                      lambda X, t: X[t, :, 1],
                      lambda X, t: X[t, :, 2] - torch.sigmoid(f1(X[t, :, [0,1]]).flatten()),
                      lambda X, t: X[t, :, 3] - (0.5*X[t-1, :, 3] + f2(X[t, :, [0,1,2]]).flatten()),
                      ]
        
class LearnedCOMPASTemporal(SemiSyntheticCOMPASTemporal):

    def __init__(self, model_type="linear", seed=2024):
        super().__init__(0, 0, 0, False, seed)

        self._generator = np.random.default_rng(2024)
        self.model_type = model_type

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
        ]

        self.U = [
            Poisson(1, seed=seed),
            Binomial(1, 0.8, seed=seed+42),
            Gaussian(0, 1, seed=seed+95),
            Gaussian(0, 1, seed=seed+11),
        ]
    
    def get_eqs(self):
        if self.model_type == "linear":
            return torch.nn.Linear(4, 1), torch.nn.Linear(5, 1)
        elif self.model_type == "dnn":
            return MLP1(4, 10), MLP1(5, 10)
        else:
            return CVAE(4), CVAE(5)

    def fit_eqs(self, Xn, output_name="linear", path=".", force_train=False):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = 'lin' if self.model_type == "linear" else 'mlp'
        if os.path.isfile(os.path.join(path, f"{output_name}_{model_type}_f1.pth")) and not force_train:
            print('Fitted SCM already exists')
            return
        
        print("Fitting SCM equations")

        # [0,1,2,3, 4,5,6,7, 8]
        mask_1 = [2,4,5,8]
        mask_2 = [3,4,5,6,8] # x_1^t-1, x_0^t, t

        f1, f2= self.get_eqs()

        # Re-build data from the originals
        X = self.Xn2X(torch.Tensor(Xn))

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
        indices = self._generator.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]
        id_train = indices

        train_epochs = 10
        if self.model_type != "cvae":
            trainer = SCM_Trainer(verbose=True, print_freq=1, lr=0.005, batch_size=N_train//100)     
        else:
            trainer = CVAETrainer()   
        
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 6].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 6].reshape(-1, 1), train_epochs)
        
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 7].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 7].reshape(-1, 1), train_epochs)

        # Save the structural equations
        if path is not None:
            torch.save(f1.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f1.pth"))
            torch.save(f2.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f2.pth"))

        self.set_eqs(f1, f2) # Build the structural equations
    
    def load(self, output_name="compas_1.0_linear", path="./learned_scms/compas"):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2 = self.get_eqs()

        model_type = 'lin' if self.model_type == "linear" else 'mlp'
        f1.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f1.pth"), weights_only=True))
        f2.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f2.pth"), weights_only=True))

        self.set_eqs(f1, f2)

    def set_eqs(self, f1, f2):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1 = f1
        self.f2 = f2

        self.f = [lambda X, U1, t: X[t-1, :, 0] + (t==1)*U1,
                  lambda X, U2, t: X[t-1, :, 1] + (t==1)*U2,
                  lambda X, U3, t: f1(torch.cat([X[t-1, :, [2]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * t], 1)).flatten() + U3,
                  lambda X, U4, t: f2(torch.cat([X[t-1, :, [3]], X[t, :, [0,1,2]], torch.ones((X.shape[1], 1)) * t], 1)).flatten() + U4,
                  ]

        self.inv_f = [lambda X, t: X[t, :, 0],
                      lambda X, t: X[t, :, 1],
                      lambda X, t: X[t, :, 2] - f1(torch.cat([X[t-1, :, [2]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * t], 1)).flatten(),
                      lambda X, t: X[t, :, 3] - f2(torch.cat([X[t-1, :, [3]], X[t, :, [0,1,2]], torch.ones((X.shape[1], 1)) * t], 1)).flatten(),
                      ]

class SemiSyntheticLoanTemporal(SCM):
    """ Semi-synthetic SCM for a loan application based on Karimi et al., 2021. """
    def __init__(self, alpha=0.01, beta_linear=1.0, beta_seasonal=0.0, seed=2024, **kwargs):
        super().__init__(**kwargs)

        # Components needed for the trend
        self.alpha = alpha # Strength of the trend component
        self.beta1 = beta_linear # Strength of the linear trend
        self.beta2 = beta_seasonal # Strength of the seasonal trend

        self.U = [
            Binomial(1, 0.5, seed=seed),
            Gamma(10, 3.5, seed=seed+3),
            Gaussian(0, np.sqrt(0.25), seed=seed+1),
            Gaussian(0, 2, seed=seed+2),
            Gaussian(0, 3, seed=seed+3),
            Gaussian(0, 2, seed=seed+4),
            Gaussian(0, 5, seed=seed+5),
        ]

        self.f = [
                    lambda X, U1, t: X[t-1, :, 0] + (t==1)*U1,
                    lambda X, U2, t: 0.5*X[t-1, :, 1] + (-35 + U2), # D x N
                    lambda X, U3, t: 0.5*X[t-1, :, 2] -0.5 + 1/(1+torch.exp(
                        -(-1 +0.5*X[t, :, 0]+1/(1+torch.exp(-0.1*X[t, :, 1]))+U3)
                    )),
                    lambda X, U4, t: 0.5*X[t-1, :, 3] + 1 + 0.01*(X[t, :, 1]-5)*(5-X[t, :, 1]) + X[t, :, 0] + U4,
                    lambda X, U5, t: 0.5*X[t-1, :, 4] + -1 + 0.1*X[t, :, 1] + 2*X[t, :, 0]+X[t, :, 3] + U5,
                    lambda X, U6, t: 0.5*X[t-1, :, 5] + -4 + 0.1*(X[t, :, 1]+35) + 2*X[t, :, 0] + X[t, :, 0]*X[t, :, 2] + U6,
                    lambda X, U7, t: 0.5*X[t-1, :, 6] + -4 +1.5*(X[t, :, 5] > 0)*X[t, :, 5] + U7
                ]

        self.inv_f = [
                        lambda X, t: X[t, :, 0],#(t==1)*X[t, :, 0] + X[t-1, :, 0],
                        lambda X, t: X[t, :, 1] - 0.5*X[t-1, :, 1] + 35, 
                        lambda X, t: - (-1 +0.5*X[t, :, 0] + 1/(1+torch.exp(-0.1*X[t, :, 1]))) - torch.log(1/(X[t, :, 2]+0.5-0.5*X[t-1, :, 2])-1),
                        lambda X, t: X[t, :, 3] - (0.5*X[t-1, :, 3] + 1 + 0.01*(X[t, :, 1]-5)*(5-X[t, :, 1]) + X[t, :, 0]),
                        lambda X, t: X[t, :, 4] - (0.5*X[t-1, :, 4] + -1 + 0.1*X[t, :, 1] + 2*X[t, :, 0]+X[t, :, 3]),
                        lambda X, t: X[t, :, 5] - (0.5*X[t-1, :, 5] + -4 + 0.1*(X[t, :, 1]+35) + 2*X[t, :, 0] + X[t, :, 0]*X[t, :, 2]),
                        lambda X, t: X[t, :, 6] - (0.5*X[t-1, :, 6] + -4 +1.5*(X[t, :, 5] > 0)*X[t, :, 5])
                    ]
        
        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]) - self.alpha * (self.beta1 * min(0.5*t, 100) + self.beta2 * np.abs(np.sin(0.5*t))),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X - (torch.zeros_like(X[0]) - self.alpha * (self.beta1 * min(0.5*t, 100) + self.beta2 * np.abs(np.sin(0.5*t)))),
            lambda X, t: X,
        ]

        # Mean and std computed at time 0
        self.mean = torch.Tensor([0, 0.07906583, -0.05792204, 0.03228487, 0.08026928, 0.49916026, -2.1743872])
        self.std = torch.Tensor([1, 11.074237, 0.13772593, 2.787965, 4.545642, 2.5124693, 5.564847])

        self.actionable = [0,1,2]
        self.soft_interv = [True, True, True]

    def sample_U(self, N, T, S=1):
        U1 = np.tile(self.U[0].sample((S,1,N)), (1, T, 1))
        U2 = self.U[1].sample((S,T,N))
        U3 = self.U[2].sample((S,T,N))
        U4 = self.U[3].sample((S,T,N))
        U5 = self.U[4].sample((S,T,N))
        U6 = self.U[5].sample((S,T,N))
        U7 = self.U[6].sample((S,T,N))

        return np.stack([U1, U2, U3, U4, U5, U6, U7], axis=-1) # T x N x D
    
    def get_descendant_mask(self, actionable):
        N, D = actionable.shape
        descendant_mask = torch.ones(N, D)
        for i in range(D):
            if actionable[0, i] == 1:
                if i == 0:
                    descendant_mask[:, [0, 2, 3, 4, 5]] = 0
                elif i == 1:
                    descendant_mask[:, [1, 2, 3, 4, 5]] = 0
                elif i == 2:
                    descendant_mask[:, [2, 5]] = 0
                elif i == 3:
                    descendant_mask[:, [3]] = 0
                elif i == 4:
                    descendant_mask[:, [4]] = 0
                elif i == 5:
                    descendant_mask[:, [5, 6]] = 0
                elif i == 6:
                    descendant_mask[:, [6]] = 0
        return descendant_mask

    def label(self, X):
        _generator = np.random.default_rng(2024)
        L, D, I, S = X[:, :, 3], X[:, :, 4], X[:, :, 5], X[:, :, 6]
        p = 1 / (1 + np.exp(-0.3*(-L -D + I + S)))
        Y = _generator.binomial(1, p)
        return Y

class LearnedLoanTemporal(SemiSyntheticLoanTemporal):
    
    def __init__(self, linear=False, time_preprocess="linear", seed=2024):
        
        super().__init__(0, 0, 0, seed)

        self._generator = np.random.default_rng(2024)
        self._time_preprocess = time_preprocess
        self.linear = linear

        self.U = [
            Gaussian(0, 0, seed=seed+11), # We assume these are fixed, taken from the data so no randomness
            Gamma(10, 3.5, seed=seed+3),
            Gaussian(0, 1, seed=seed+13),
            Gaussian(0, 1, seed=seed+12),
            Gaussian(0, 1, seed=seed+3),
            Gaussian(0, 1, seed=seed+4),
            Gaussian(0, 1, seed=seed+1),
            Gaussian(0, 1, seed=seed+2),
        ]

        self.trend = [
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
            lambda X, t: torch.zeros_like(X[t, :, 0]),
        ]

        # Dimensions here are different since we compute it for a slice only
        self.inv_trend = [
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X,
            lambda X, t: X
        ]
    
    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(4, 1), torch.nn.Linear(4, 1), torch.nn.Linear(5, 1), torch.nn.Linear(5, 1), torch.nn.Linear(3, 1)
        return MLP1(4, 10), MLP1(4, 10), MLP1(5, 10), MLP1(5, 10), MLP1(3, 10)
    
    def process_time(self, t):
        if self._time_preprocess == "linear":
            return t
        elif self._time_preprocess == "sin":
            return np.sin(t)
        return t 

    def fit_eqs(self, Xn, output_name="linear", path=".", force_train=False):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = 'lin' if self.linear else 'mlp'
        if os.path.isfile(os.path.join(path, f"{output_name}_{model_type}_f1.pth")) and not force_train:
            print('Fitted SCM already exists')
            return
        
        print("Fitting SCM equations")

        # [0,1,2,3,4,5,6,  7,8,9,10,11,12,13, 14]
        mask_1 = [2, 7, 8, 14]
        mask_2 = [3, 7, 8, 14] # x_1^t-1, x_0^t, t
        mask_3 = [4, 7, 8, 10, 14]
        mask_4 = [5, 7, 8, 9, 14] # x_1^t-1, x_0^t, t
        mask_5 = [6, 12, 14]

        f1, f2, f3, f4, f5 = self.get_eqs()

        # Re-build data from the originals
        X = self.Xn2X(torch.Tensor(Xn))

        # Regenerate the data such to be in the correct format to train
        # the structural equations
        rearranged_data = []
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                 if t == 0:
                    rearranged_data.append(
                        np.zeros_like(X[t, i, :]).tolist()+X[t, i, :].tolist()+[self.process_time(t)]
                    )
                 else:
                    rearranged_data.append(
                        X[t-1, i, :].tolist()+X[t, i, :].tolist()+[self.process_time(t)]
                    )
        X = torch.Tensor(rearranged_data)

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = self._generator.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]
        id_train = indices

        train_epochs = 10
        trainer = SCM_Trainer(verbose=True, print_freq=1, lr=0.005, batch_size=N_train//100)        
        
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 9].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 9].reshape(-1, 1), train_epochs)
        
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 10].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 10].reshape(-1, 1), train_epochs)
        
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 11].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 11].reshape(-1, 1), train_epochs)
        
        trainer.train(f4, X[id_train][:, mask_4], X[id_train, 12].reshape(-1, 1),
                      X[id_test][:, mask_4], X[id_test, 12].reshape(-1, 1), train_epochs)
        
        trainer.train(f5, X[id_train][:, mask_5], X[id_train, 13].reshape(-1, 1),
                      X[id_test][:, mask_5], X[id_test, 13].reshape(-1, 1), train_epochs)

        # Save the structural equations
        if path is not None:
            torch.save(f1.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f1.pth"))
            torch.save(f2.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f2.pth"))
            torch.save(f3.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f3.pth"))
            torch.save(f4.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f4.pth"))
            torch.save(f5.state_dict(), os.path.join(path, f"{output_name}_{model_type}_f5.pth"))

        self.set_eqs(f1, f2, f3, f4, f5) # Build the structural equations

    def load(self, output_name="loan_1.0_linear", path="./learned_scms/loan"):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2, f3, f4, f5 = self.get_eqs()

        model_type = 'lin' if self.linear else 'mlp'
        f1.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f1.pth"), weights_only=True))
        f2.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f2.pth"), weights_only=True))
        f3.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f3.pth"), weights_only=True))
        f4.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f4.pth"), weights_only=True))
        f5.load_state_dict(torch.load(os.path.join(path, f"{output_name}_{model_type}_f5.pth"), weights_only=True))

        self.set_eqs(f1, f2, f3, f4, f5)

    def _btsp(self, X, current_feature, parents, time):
        """_build_tensor_for_prediction """
        return torch.cat(
            [   
                X[time-1, :, current_feature],
                X[time, :, parents],
                torch.ones((X.shape[1], 1)) * self.process_time(time)
            ], 1)

    def set_eqs(self, f1, f2, f3, f4, f5):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1, self.f2, self.f3, self.f4, self.f5 = f1, f2, f3, f4, f5

        self.f = [lambda X, U1, t: X[t-1, :, 0] + (t==1)*X[t, :, 0],
                  lambda X, U2, t: 0.5*X[t-1, :, 1] + (-35 + U2),
                  lambda X, U3, t: f1(self._btsp(X, [2], [0,1], t)).flatten() + U3,
                  lambda X, U4, t: f2(self._btsp(X, [3], [0,1], t)).flatten() + U4,
                  lambda X, U5, t: f3(self._btsp(X, [4], [0,1,3], t)).flatten() + U5,
                  lambda X, U6, t: f4(self._btsp(X, [5], [0,1,2], t)).flatten() + U6,
                  lambda X, U7, t: f5(self._btsp(X, [6], [5], t)).flatten() + U7,
                  ]

        self.inv_f = [lambda X, t: X[t, :, 0],
                      lambda X, t: X[t, :, 1] - 0.5*X[t-1, :, 1] + 35,
                      lambda X, t: X[t, :, 2] - f1(torch.cat([X[t-1, :, [2]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 3] - f2(torch.cat([X[t-1, :, [3]], X[t, :, [0,1]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 4] - f3(torch.cat([X[t-1, :, [4]], X[t, :, [0,1,3]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 5] - f4(torch.cat([X[t-1, :, [5]], X[t, :, [0,1,2]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      lambda X, t: X[t, :, 6] - f5(torch.cat([X[t-1, :, [6]], X[t, :, [5]], torch.ones((X.shape[1], 1)) * self.process_time(t)], 1)).flatten(),
                      ]