import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class VAE_nas301(nn.Module):
    def __init__(self, INPUT_DIM, LATENT_DIM, output_shape=(2, 7, 6, 6)):
        super().__init__()

        self.INPUT_DIM = INPUT_DIM
        self.LATENT_DIM = LATENT_DIM
        self.output_shape = output_shape
        self.output_dim = int(torch.prod(torch.tensor(output_shape)).item())

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, LATENT_DIM)
        self.logvar = nn.Linear(128, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, INPUT_DIM)
        )

        self.acc_predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z

    def decode(self, z):
        recon_logits = self.decoder(z)
        recon_logits = recon_logits.view(z.size(0), *self.output_shape)

        recon_probs = torch.sigmoid(recon_logits)

        return recon_logits, recon_probs

    def predict_acc(self, z):
        return self.acc_predictor(z).squeeze(-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        recon_logits, recon_probs = self.decode(z)

        pred_acc = self.predict_acc(z)

        return recon_logits, recon_probs, mu, logvar, pred_acc

def vae_accuracy_loss_nas301(
    recon_logits,
    recon_probs,
    x,
    mu,
    logvar,
    acc_pred,
    true_acc,
    beta=0.0,
    lambda_acc=1.0,
    pos_weight_value=5.0
):
    x = x.view(x.size(0), 2, 7, 6, 6)
    bce = F.binary_cross_entropy_with_logits(
        recon_logits,
        x,
        reduction="none"
    )
    #weights on ones since the matrix is very sparse so the VAE could learn to reconstruct just the zeros
    weights = torch.ones_like(x)
    weights[x > 0.5] = pos_weight_value
    recon_loss = (bce * weights).mean()

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    true_acc = true_acc.view(-1).float()
    acc_pred = acc_pred.view(-1).float()
    acc_loss = F.mse_loss(
        acc_pred,
        true_acc
    )

    total_loss = recon_loss + beta * kl + lambda_acc * acc_loss

    return total_loss, recon_loss, kl, acc_loss
