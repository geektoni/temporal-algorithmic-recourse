import torch
import numpy as np

from sklearn.metrics import accuracy_score

#
#
# Original code taken from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/blob/main/scm.py
#
#

class LogisticRegression(torch.nn.Module):
    """ Logistic regression model for classification/regression"""
    def __init__(self, input_size):
        """
        Inputs:     input_size: int, number of features of the data
                    hidden_size: int, number of neurons for the hidden layer
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, 1, dtype=torch.float32)

    def forward(self, x):
        """
        Inputs:     x: torch.Tensor, shape (N, input_size)

        Outputs:    torch.Tensor, shape (N, 2)
        """
        return self.linear1(x).flatten()

    def predict_torch(self, x, flatten=True):
        if flatten:
            return torch.where(self.linear1(x) > 0.5, 1.0, 0.0)
        else:
            return torch.sigmoid(self.linear1(x))
    

class MLP1(torch.nn.Module):
    """ MLP with 3-layer and tanh activation function"""
    def __init__(self, input_size, hidden_size=100):
        """
        Inputs:     input_size: int, number of features of the data
                    hidden_size: int, number of neurons for the hidden layer
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.linear3 = torch.nn.Linear(hidden_size, 1, dtype=torch.float32)
        self.activ = torch.nn.Tanh()

    def forward(self, x):
        """
        Inputs:     x: torch.Tensor, shape (T, N, input_size)

        Outputs:    torch.Tensor, shape (T, N, 2)
        """
        return self.linear3(self.activ(self.linear2(self.activ(self.linear1(x))))).flatten()

    def predict_torch(self, x, flatten=True):
        if flatten:
            return torch.where(self.linear3(self.activ(self.linear2(self.activ(self.linear1(x))))) > 0.5, 1.0, 0.0)
        else:
            return torch.sigmoid(self.linear3(self.activ(self.linear2(self.activ(self.linear1(x))))))
    

class MLPTrainer:
    """ Class used to fit the structural equations of some SCM """
    def __init__(self, batch_size=100, lr=0.001, print_freq=100, verbose=False):
        """
        Inputs:     batch_size: int
                    lr: float, learning rate (Adam used as the optimizer)
                    print_freq: int, verbose every print_freq epochs
                    verbose: bool
        """
        self.batch_size = batch_size
        self.lr = lr
        self.print_freq = print_freq
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.verbose = verbose

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs):
        """
        Inputs:     model: torch.nn.Model
                    X_train: torch.Tensor, shape (N, D)
                    Y_train: torch.Tensor, shape (N, 1)
                    X_test: torch.Tensor, shape (M, D)
                    Y_test: torch.Tensor, shape (M, 1)
                    epochs: int, number of training epochs
        """
        X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
        X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)
        
        train_dst = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            if self.verbose:
                if epoch % self.print_freq == 0:
                    mse = accuracy_score(torch.where(model(X_test) > 0.5, 1.0, 0).numpy(), Y_test.numpy())
                    print("Epoch: {}. MSE {}.".format(epoch, mse))

            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self.loss_function(model(x), y)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()