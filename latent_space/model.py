import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 16
INPUT_DIM = 96


class VAE(nn.Module):

    def __init__(self):

        super().__init__()


        self.fc1 = nn.Linear(INPUT_DIM, 128)
        self.fc2 = nn.Linear(128, 64)

        self.mu = nn.Linear(64, LATENT_DIM)
        self.logvar = nn.Linear(64, LATENT_DIM)


        self.fc3 = nn.Linear(LATENT_DIM, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, INPUT_DIM)


        self.acc_predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def predict_acc(self, z):
        return self.acc_predictor(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        acc_pred = self.predict_acc(z)

        return recon, mu, logvar, acc_pred

def vae_loss(recon, x, mu, logvar,beta=1):

    # ricostruzione
    recon_loss = F.mse_loss(recon, x)

    # KL divergence
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl * beta

def vae_accuracy_loss(
    recon,
    x,
    mu,
    logvar,
    pred_acc,
    true_acc,
    beta=1.0,
    lambda_acc=1.0
):

    recon_loss = F.mse_loss(recon, x)

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    acc_loss = F.mse_loss(pred_acc.squeeze(), true_acc)

    loss = recon_loss + beta * kl + lambda_acc * acc_loss

    return loss, recon_loss, kl, acc_loss

class FlowNet(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.net(x)

def flow_loss(model, x, y):
    target = y - x
    pred = model(x)
    return ((pred - target) ** 2).mean()