import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

# cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} 


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, latent_dim=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim + input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # Predict Y (scalar target)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        mu = torch.clamp(mu, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_cond):
        z_cond = torch.cat([z, x_cond], dim=1)
        h3 = torch.relu(self.fc3(z_cond))
        return self.fc4(h3)

    def forward(self, x_cond):
        mu, logvar = self.encode(x_cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x_cond), mu, logvar

    def sample(self, x_cond):
        with torch.no_grad():
            mu, logvar = self.encode(x_cond)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, x_cond)


class CVAETrainer:

    def __init__(self, batch_size=100) -> None:
        self.batch_size = batch_size

    def loss_function(self, recon_y, y, mu, logvar):
        recon_loss = nn.MSELoss()(recon_y, y)
        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss


    def train(self, model, X_train, Y_train, X_test, Y_test, lr=1e-3, epochs=15):

        X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
        X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)
        
        train_dst = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)

        with torch.autograd.set_detect_anomaly(True):

            model.train()
            for epoch in range(epochs):
                train_loss = 0
                for batch_idx, (x, y) in enumerate(train_loader):
                    
                    x, y = x.to(device), y.to(device)

                    print(x)

                    optimizer.zero_grad()
                    recon_y, mu, logvar = model(x)
                    loss = self.loss_function(recon_y, y, mu, logvar)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    
                    optimizer.step()
                    if batch_idx % 20 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(x), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item() / len(x)))

                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(train_loader.dataset)))
