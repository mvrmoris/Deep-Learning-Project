import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

NUM_NODES = 4
NUM_OPS = 5
NUM_EDGES = NUM_NODES * NUM_NODES

class VAE_dist(nn.Module):
    def __init__(self, INPUT_DIM=80, LATENT_DIM=16, output_shape=(4, 4, 5)):
        super().__init__()

        self.INPUT_DIM = INPUT_DIM
        self.LATENT_DIM = LATENT_DIM
        self.output_shape = output_shape
        self.output_dim = int(np.prod(output_shape))
        self.fc1 = nn.Linear(INPUT_DIM, 128)
        self.fc2 = nn.Linear(128, 64)

        self.mu = nn.Linear(64, LATENT_DIM)
        self.logvar = nn.Linear(64, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.edge_logits = nn.Linear(128, self.output_dim)

        self.acc_predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        h = self.decoder(z)
        logits = self.edge_logits(h)
        logits = logits.view(
            z.size(0),
            *self.output_shape
        )
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict_acc(self, z):
        return self.acc_predictor(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        recon_logits, recon_probs = self.decode(z)

        acc_pred = self.predict_acc(z)

        return recon_logits, recon_probs, mu, logvar, acc_pred
    
def vae_accuracy_loss(
    recon_logits,
    recon_probs,   
    x,
    mu,
    logvar,
    acc_pred,
    true_acc,
    beta=1.0,
    lambda_acc=1.0,
    **kwargs   
):

    x = x.reshape(
        x.size(0),
        NUM_OPS,
        NUM_NODES,
        NUM_NODES
    )

    target_onehot = x.permute(0, 2, 3, 1)
    target = target_onehot.argmax(dim=-1)
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, NUM_OPS),
        target.reshape(-1).long()
    )
    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    if lambda_acc > 0:
        acc_loss = F.mse_loss(
            acc_pred.squeeze(-1),
            true_acc.float()
        )
    else:
        acc_loss = torch.tensor(0.0, device=x.device)

    loss = recon_loss + beta * kl + lambda_acc * acc_loss

    return loss, recon_loss, kl, acc_loss


   
def vae_accuracy_loss_ws(
    recon_logits,
    recon_probs,   
    x,
    mu,
    logvar,
    acc_pred,
    true_acc,
    beta=1.0,
    **kwargs   
):

    x = x.reshape(
        x.size(0),
        NUM_OPS,
        NUM_NODES,
        NUM_NODES
    )

    target_onehot = x.permute(0, 2, 3, 1)
    target = target_onehot.argmax(dim=-1)
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, NUM_OPS),
        target.reshape(-1).long()
    )
    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    loss = recon_loss + beta * kl 

    return loss, recon_loss, kl