"""
This file contains the implementation of the Structural Causal Models used for modelling the effect of interventions
on the features of the individual seeking recourse.
"""

import os
import numpy as np
import torch

from itertools import chain, combinations, product  # for the powerset of actionable combinations of interventions

class SCM:
    """
    Includes all the relevant methods required for generating counterfactuals. Classes inheriting this class must
    contain the following objects:
        self.f: list of functions, each representing a structural equation. Function self.f[i] must have i+1 arguments,
                corresponding to X_1, ..., X_{i-1}, U_{i+1} each being a torch.Tensor with shape (N, 1), and returns
                the endogenous variable X_{i+1} as a torch.Tensor with shape (N, 1)
        self.inv_f: list of functions, corresponding to the inverse mapping X -> U. Each function self.inv_f[i] takes
                    as argument the features X as a torch.Tensor with shape (N, D), and returns the corresponding
                    exogenous variable U_{i+1} as a torch.Tensor with shape (N, 1)
        self.actionable: list of int, indices of the actionable features
        self.soft_interv: list of bool with len = D, indicating whether the intervention on feature soft_interv[i] is
                          modeled as a soft intervention (True) or hard intervention (False)
        self.mean: expectation of the features, such that when generating data we can standarize it
        self.std: standard deviation of the features, such that when generating data we can standarize it
    """
    def sample_U(self, N, T, S=1):
        """
        Return N samples from the distribution over exogenous variables P_U.

        Inputs:     N: int, number of samples to draw

        Outputs:    U: np.array with shape (T, N, D)
        """
        raise NotImplementedError

    def label(self, X):
        """
        Label the input instances X

        Inputs:     X: np.array with shape (T, N, D)

        Outputs:    Y:  np.array with shape (T, N)
        """
        raise NotImplementedError
    
    def get_descendant_mask(self, actionable):
        raise NotImplementedError

    def generate(self,
                 N: int,
                 T: int=1,
                 U: torch.tensor=None,
                 intervention: torch.tensor=None,
                 past: torch.tensor=None,
                 soft_intervention: bool=True,
                 sample_size: int=1,
                 return_sample: bool=False):
        """
        Sample from the observational distribution implied by the SCM

        Inputs:     N: int, number of instances to sample
                    T: int, time frame where to generate data
                    intervention: torch.tensor with shape (T, N, D), intervention to apply to the system
                    past: torch.tensor with shape (T_1, N, D), past values we can specify to condition the generation
                    soft_intervention: bool, if we want hard or soft interventions
                    sample_size: int (default is 1), if > 1 we compute the expected value E[X]

        Outputs:    X: np.array with shape (T, N, D), standarized (since we train the models on standarized data)
                    Y: np.array with shape (T, N)
        """

        # If the intervention is None, then we set it as empty
        intervention = intervention if intervention is not None else torch.zeros(1, N, len(self.f)) 

        # Scale the intervention appropriately
        intervention = torch.Tensor(intervention) * self.std

        # Perform multiple sampling if needed
        samples =[]
        U_sample = torch.Tensor(self.sample_U(N, T, S=sample_size).astype(np.float32)) if U is None else U
        for i in range(sample_size):
            X = self.U2X(U_sample[i, :, :, :],
                        intervention,
                        soft_intervention,
                        self.Xn2X(past) if past is not None else past)
            samples.append(X)
        
        # Take the mean of all the samples
        X = torch.mean(torch.stack(samples), 0)
        
        Y = self.label(X.detach().numpy())
        X = (X - self.mean) / self.std

        with torch.no_grad():
            if not return_sample:
                return X.detach().numpy(), Y, U_sample
            else:
                return X.detach().numpy(), Y, U_sample, torch.stack(samples).numpy()

    def U2X(self, U: torch.tensor,
            intervention: torch.tensor=None,
            soft_intervention: bool=True,
            fixed_past: torch.tensor=None
        ):
        """
        Map from the exogenous variables U to the endogenous variables X by using the structural equations self.f

        Inputs:     U: torch.Tensor with shape (T, N, D), exogenous variables
                    intervention: torch.Tensor with shape (T, N, D), interventions
                    soft_intervention: bool, if do hard or soft intervention
                    start_t: int (default is 0), time step where we can start sampling

        Outputs:    X: torch.Tensor with shape (T, N, D), endogenous variables
        """

        # We add an additional dimension to accomodate for t=0
        X = torch.zeros((U.shape[0]+1, U.shape[1], U.shape[2]))
        T = 1

        # Check if the intervention is a single time
        single_time_intervention = intervention.shape[0] == 1 if intervention is not None else False

        # If we specified a fixed past, we need to clone the corresponding
        # values into their right place.
        if fixed_past is not None:
            T = fixed_past.shape[0] + 1 # We need to add an additional unit to account for the zeroth step
            assert X.shape[0] > T, "fixed_past size is greater than the sampling one"
            X[1:T, :, :] = fixed_past.clone()

        for t in range(T, X.shape[0]):
            U_t = U[t-1,:,:] # 1 x N x D
            for i in range(U.shape[2]):
                str_equation_args = [X] + [U_t[:, i]] + [t]
                new_value = self.f[i](*str_equation_args).flatten() + self.trend[i](X, t-1).flatten()
                
                # This step is needed to avoid weird errors caused
                # by autograd compilations
                # https://discuss.pytorch.org/t/torch-exp-is-modified-by-an-inplace-operation/90216
                X = X.clone()
                
                X[t, :, i] = new_value # 1 x N x 1

                # Either additive soft intervention or hard one
                if intervention is not None:
                    if soft_intervention:
                        
                        # Apply only the last element
                        if single_time_intervention:
                            if t == X.shape[0]-1:
                                X[t, :, i] += intervention[0, :, i]
                        else:
                            X[t, :, i] += intervention[t-1, :, i]
                    else:
                        assert False
                        X[t, :, i] = intervention[t-1, :, i]

        # We return the value by skipping the first element (since it is all zeros)
        return X[1:, :, :]

    def X2U(self, X):
        """
        Map from the endogenous variables to the exogenous variables by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (T, N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (T, N, D), endogenous variables
        """
        if self.inv_f is None:
            return X + 0.
        T,N,D = X.shape
        U = torch.zeros_like(X)
        X_ext = torch.concatenate((torch.zeros((1, N, D)), X), axis=0)
        for t in range(1, X_ext.shape[0]):
            for i in range(X_ext.shape[2]):
                U[t-1, :, i] = self.inv_trend[i](self.inv_f[i](X_ext, t), t-1)
        return U

    def counterfactual(self, Xn, delta=None, actionable=None, soft_interv=True, return_U=False):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (T, N, D) factual
                    delta: torch.Tensor (T, N, D), intervention values
                    actionable: None or list of int, indices of the intervened upon variables
                    soft_interv: None or list of int, variables for which the interventions are soft (rather than hard)
                    return_U: bool (default is false), return also the exogenous variables obtained via abduction

        Outputs:
                    X_cf: torch.Tensor (T, N, D), counterfactual
        """

        delta = torch.zeros_like(Xn) if delta is None else delta
        actionable = torch.ones_like(delta) if actionable is None else actionable

        # Mask the intervention
        delta = delta * actionable

        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        if return_U:
            return self.X2Xn(self.U2X(U, delta, soft_interv)), U
        else:
            return self.X2Xn(self.U2X(U, delta, soft_interv))
    
    def interventional(self, Xn, delta=None, actionable=None, soft_interv=True, sample=1, return_samples=False):
        
        T, N, D = Xn.shape

        delta = np.zeros_like(Xn) if delta is None else delta
        actionable = torch.ones((N, D)) if actionable is None else actionable
        
        # Mask the intervention
        delta = delta * actionable

        # Scale appropriately
        delta = delta * self.std

        # We want to condition on the non-descendant of the intervention
        # such to keep those values
        descendant_mask = self.get_descendant_mask(actionable)

        # Copy everything
        X = self.Xn2X(Xn)

        # Simulate the exogenous factors and copy the previous values
        U = torch.FloatTensor(self.sample_U(N, 1, S=sample))

        # Simulate the users for all time steps
        samples = []
        for idx in range(sample):
            Xtmp = X.clone()
            U_t = U[idx,0,:,:] # 1 x N x D
            for i in range(D):
                str_equation_args = [Xtmp] + [U_t[:, i]] + [T-1]
                new_value = Xtmp[T-1, :, i]*descendant_mask[:, i] + (1-descendant_mask[:, i]) * (self.f[i](*str_equation_args).flatten() + self.trend[i](Xtmp, T-1).flatten()) # 1 x N x 1

                # This step is needed to avoid weird errors caused
                # by autograd compilations
                # https://discuss.pytorch.org/t/torch-exp-is-modified-by-an-inplace-operation/90216
                Xtmp = Xtmp.clone()
                Xtmp[T-1, :, i] = new_value

                if soft_interv:
                    Xtmp[T-1, :, i] += delta[0, :, i]
                else:
                    assert False
            samples.append(self.X2Xn(Xtmp))

        # Return the empirical expected value
        if return_samples:
            return torch.stack(samples)
        else:
            return torch.mean(torch.stack(samples), 0)
    
    def interventional_batch(self, Xn, delta=None, actionable=None, soft_interv=True):
        
        S, T, N, D = Xn.shape

        delta = np.zeros_like(Xn) if delta is None else delta
        actionable = torch.ones((N, D)) if actionable is None else actionable
        
        # Mask the intervention
        delta = delta * actionable

        # We want to condition on the non-descendant of the intervention
        # such to keep those values
        descendant_mask = self.get_descendant_mask(actionable)

        # Simulate the exogenous factors and copy the previous values
        U = torch.FloatTensor(self.sample_U(N, 1, S=S))

        # Scale delta appropriately
        delta = delta * self.std

        # Standardize the data
        Xn = self.Xn2X(Xn)

        # Simulate the users for all time steps
        X = Xn.clone()
        for idx in range(S):
            U_t = U[idx, 0,:,:] # 1 x N x D
            for i in range(D):
                str_equation_args = [X[idx, :, :, :]] + [U_t[:, i]] + [T-1]
                new_value = X[idx, T-1, :, i]*descendant_mask[:, i] + (1-descendant_mask[:, i]) * (self.f[i](*str_equation_args).flatten() + self.trend[i](X[idx, :, :, :], T-1).flatten()) # 1 x N x 1

                # This step is needed to avoid weird errors caused
                # by autograd compilations
                # https://discuss.pytorch.org/t/torch-exp-is-modified-by-an-inplace-operation/90216
                X = X.clone()
                X[idx, T-1, :, i] = new_value

                if soft_interv:
                    X[idx, T-1, :, i] += delta[0, :, i]
                else:
                    assert False
        return self.X2Xn(X)        


    def counterfactual_batch(self, Xn, delta, interv_mask):
        """
        Inputs:     Xn: torch.Tensor (T, N, D) factual
                    delta: torch.Tensor (T, N, D), intervention values
                    interv_sets: torch.Tensor (T, N, D)

        Outputs:
                    X_cf: torch.Tensor (T, N, D), counterfactual
        """
        T, N, D = Xn.shape
        soft_mask = torch.Tensor(self.soft_interv).repeat(N, 1)
        hard_mask = 1. - soft_mask

        mask_hard_actionable = hard_mask * interv_mask
        mask_soft_actionable = soft_mask * interv_mask

        return self.counterfactual_mask(Xn, delta, mask_hard_actionable, mask_soft_actionable)


    def counterfactual_mask(self, Xn, delta, mask_hard_actionable, mask_soft_actionable):
        """
        Different way of computing counterfactuals, which may be more computationally efficient in some cases, specially
        if different instances have different actionability constrains, or hard/soft intervention criteria.

        Inputs:     Xn: torch.Tensor (T, N, D) factual
                    delta: torch.Tensor (T, N, D), intervention values
                    mask_hard_actionable: torch.Tensor (T, N, D), 1 for actionable features under a hard intervention
                    mask_soft_actionable: torch.Tensor (T, N, D), 1 for actionable features under a soft intervention

        Outputs:
                    X_cf: torch.Tensor (T, N, D), counterfactual
        """
        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            X_cf.append((X[:, [i]] + delta[:, [i]]) * mask_hard_actionable[:, [i]] + (1 - mask_hard_actionable[:, [i]])
                        * (self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]] * mask_soft_actionable[:, [i]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def U2Xn(self, U):
        """
        Mapping from the exogenous variables U to the endogenous X variables, which are standarized

        Inputs:     U: torch.Tensor, shape (N, D)

        Outputs:    Xn: torch.Tensor, shape (N, D), is standarized
        """
        return self.X2Xn(self.U2X(U))

    def Xn2U(self, Xn):
        """
        Mapping from the endogenous variables X (standarized) to the exogenous variables U

        Inputs:     Xn: torch.Tensor, shape (N, D), endogenous variables (features) standarized

        Outputs:    U: torch.Tensor, shape (N, D)
        """
        return self.X2U(self.Xn2X(Xn))

    def Xn2X(self, Xn):
        """
        Transforms the endogenous features to their original form (no longer standarized)

        Inputs:     Xn: torch.Tensor, shape (N, D), features are standarized

        Outputs:    X: torch.Tensor, shape (N, D), features are not standarized
        """
        return Xn * self.std + self.mean

    def X2Xn(self, X):
        """
        Standarizes the endogenous variables X according to self.mean and self.std

        Inputs:     X: torch.Tensor, shape (N, D), features are not standarized

        Outputs:    Xn: torch.Tensor, shape (N, D), features are standarized
        """
        return (X - self.mean) / self.std

    def getActionable(self):
        """ Returns the indices of the actionable features, as a list of ints. """
        return self.actionable

    def getPowerset(self, actionable, when_to_apply: int):
        """ Returns the power set of the set of actionable features, as a list of lists of ints. """
        
        # Create all the potential combinations of actions by taking only one action
        # for each feature.
        all_combinations = []
        for k in range(1, len(actionable) + 1):
            for combination in combinations(actionable, k):
                all_combinations.append(combination)

        # Extend all the combinations with the timing information
        extended_all_combinations = []
        for intervention in all_combinations:
            extended_intervention = []
            for i in intervention:
                extended_intervention.append((when_to_apply,i,True))
            extended_all_combinations.append(extended_intervention)

        return list(extended_all_combinations)

    def build_mask(self, mylist, shape):
        """
        Builds a torch.Tensor mask according to the list of indices contained in mylist. Used to build the masks of
        actionable features, or those of variables which are intervened upon with soft interventions.

        Inputs:     mylist: list(D) of ints or list(N) of lists(D) of ints, corresponding to indices
                    shape: list of ints [N, D]

        Outputs:    mask: torch.Tensor with shape (N, D), where mask[i, j] = 1. if j in mylist (for list of ints) or
                          j in mylist[i] (for list of list of ints)
        """
        mask = torch.zeros(shape)
        if type(mylist[0]) == list: # nested list
            for i in range(len(mylist)):
                mask[i, mylist[i]] = 1.
        else:
            mask[:, mylist] = 1.
        return mask

    def get_masks(self, actionable, shape):
        """
        Returns the mask of actionable features, actionable features which are soft intervened, and actionable
        features which are hard intervened.

        Inputs:     actionable: list(D) of int, or list(N) of list(D) of int, containing the indices of actionable feats
                    shape: list of int [N, D]

        Outputs:    mask_actionable: torch.Tensor (N, D)
                    mask_soft_actionable: torch.Tensor (N, D)
                    mask_hard_actionable: torch.Tensor (N, D)
        """
        mask_actionable = self.build_mask(actionable, shape)
        mask_soft = self.build_mask(list(np.where(self.soft_interv)[0]), shape)
        mask_hard_actionable = (1 - mask_soft) * mask_actionable
        mask_soft_actionable = mask_soft * mask_actionable
        return mask_actionable, mask_soft_actionable, mask_hard_actionable

# ----------------------------------------------------------------------------------------------------------------------
# The following functions are to fit the structural equations using MLPs with 1 hidden layer, in the case where the
# causal graph is know but the structural equations are unknown.
# ----------------------------------------------------------------------------------------------------------------------

class MLP1(torch.nn.Module):
    """ MLP with 1-layer and tanh activation function, to fit each of the structural equations """
    def __init__(self, input_size, hidden_size=100, sigmoid=False):
        """
        Inputs:     input_size: int, number of features of the data
                    hidden_size: int, number of neurons for the hidden layer
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)
        self.activ = torch.nn.Tanh()
        self._sigmoid = sigmoid

    def forward(self, x):
        """
        Inputs:     x: torch.Tensor, shape (N, input_size)

        Outputs:    torch.Tensor, shape (N, 1)
        """
        return self.linear2(self.activ(self.linear1(x))) if not self._sigmoid else torch.sigmoid(self.linear2(self.activ(self.linear1(x)))) > 0.5


class SCM_Trainer:
    """ Class used to fit the structural equations of some SCM """
    def __init__(self, batch_size=100, lr=0.001, print_freq=100, verbose=False, seed=2024):
        """
        Inputs:     batch_size: int
                    lr: float, learning rate (Adam used as the optimizer)
                    print_freq: int, verbose every print_freq epochs
                    verbose: bool
        """
        self.batch_size = batch_size
        self.lr = lr
        self.print_freq = print_freq
        self.loss_function = torch.nn.MSELoss(reduction='mean') # Fit using the Mean Square Error
        self.verbose = verbose

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs):
        """
        Inputs:     model: torch.nn.Model
                    X_train: torch.Tensor, shape (N, D)
                    Y_train: torch.Tensor, shape (N, 1)
                    X_test: torch.Tensor, shape (M, D)
                    Y_test: torch.Tensor, shape (M, 1)
                    epochs: int, number of training epochs
        """
        X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test)
        train_dst = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        test_dst = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True,
                                                   generator=self._generator)
        test_loader = torch.utils.data.DataLoader(dataset=test_dst, batch_size=1000, shuffle=False,
                                                  generator=self._generator)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        accuracies = np.zeros(int(epochs / self.print_freq))

        prev_loss = np.inf
        val_loss = 0
        for epoch in range(epochs):
            if self.verbose:
                if epoch % self.print_freq == 0:
                    mse = self.loss_function(model(X_test), Y_test)
                    print("Epoch: {}. MSE {}.".format(epoch, mse))

            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self.loss_function(model(x), y)
                loss.backward()
                optimizer.step()


