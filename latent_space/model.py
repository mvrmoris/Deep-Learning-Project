import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_NODES = 4
NUM_OPS = 6
NUM_EDGES = NUM_NODES * NUM_NODES


class VAE_nas301(nn.Module):
    def __init__(self, INPUT_DIM, LATENT_DIM, output_shape=(2, 7, 6, 6)):
        super().__init__()

        self.INPUT_DIM = INPUT_DIM
        self.LATENT_DIM = LATENT_DIM
        self.output_shape = output_shape
        self.output_dim = int(torch.prod(torch.tensor(output_shape)).item())

        assert self.output_dim == INPUT_DIM, (
            f"INPUT_DIM={INPUT_DIM} ma output_shape produce {self.output_dim}. "
            f"Controlla output_shape."
        )

        # --------------------
        # Encoder più profondo
        # --------------------
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

        # --------------------
        # Decoder più profondo
        # --------------------
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

        # --------------------
        # Accuracy predictor
        # --------------------
        self.acc_predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x):
        """
        x: [batch, 504]
        """
        x = x.view(x.size(0), -1)
        h = self.encoder(x)

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z

    def decode(self, z):
        """
        Ritorna sia logits sia probabilità.

        logits: [batch, 2, 7, 6, 6]
        probs:  [batch, 2, 7, 6, 6]
        """
        recon_logits = self.decoder(z)
        recon_logits = recon_logits.view(z.size(0), *self.output_shape)

        recon_probs = torch.sigmoid(recon_logits)

        return recon_logits, recon_probs

    def predict_acc(self, z):
        return self.acc_predictor(z).squeeze(-1)

    def forward(self, x):
        """
        Output compatibile con:
        recon_logits, recon_probs, mu, logvar, pred_acc = model_VAE(x)
        """
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


class VAE_dist(nn.Module):

    def __init__(
        self,
        INPUT_DIM=96,
        LATENT_DIM=16,
        output_shape=(4, 4, 6)
    ):
        super().__init__()

        self.INPUT_DIM = INPUT_DIM
        self.LATENT_DIM = LATENT_DIM
        self.output_shape = output_shape
        self.output_dim = int(np.prod(output_shape))

        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, 128)
        self.fc2 = nn.Linear(128, 64)

        self.mu = nn.Linear(64, LATENT_DIM)
        self.logvar = nn.Linear(64, LATENT_DIM)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.edge_logits = nn.Linear(128, self.output_dim)

        # Predictor accuracy
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

        # softmax sull'ultima dimensione
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
    recon_probs,   # viene passato da pretrain_and_freeze_vae, anche se qui non lo usiamo
    x,
    mu,
    logvar,
    acc_pred,
    true_acc,
    beta=1.0,
    lambda_acc=1.0,
    **kwargs       # utile per ignorare eventuali argomenti extra
):
    """
    Loss VAE per NAS201.

    recon_logits: [batch, 4, 4, 6]
    recon_probs:  [batch, 4, 4, 6], non usato qui
    x:            [batch, 96]
    mu:           [batch, latent_dim]
    logvar:       [batch, latent_dim]
    acc_pred:     [batch, 1]
    true_acc:     [batch]
    """

    # x arriva appiattita: [batch, 96]
    # la riportiamo alla codifica one-hot originale: [batch, 6, 4, 4]
    x = x.reshape(
        x.size(0),
        NUM_OPS,
        NUM_NODES,
        NUM_NODES
    )

    # spostiamo le operazioni in ultima posizione: [batch, 4, 4, 6]
    target_onehot = x.permute(0, 2, 3, 1)

    # target discreto: [batch, 4, 4]
    target = target_onehot.argmax(dim=-1)

    # reconstruction loss categoriale
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, NUM_OPS),
        target.reshape(-1).long()
    )

    # KL divergence
    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # accuracy prediction loss
    if lambda_acc > 0:
        acc_loss = F.mse_loss(
            acc_pred.squeeze(-1),
            true_acc.float()
        )
    else:
        acc_loss = torch.tensor(0.0, device=x.device)

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

@torch.no_grad()
def check_reconstruction(model, loader, device="cpu"):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device).float()

        # x: [batch, 96] oppure [batch, 6, 4, 4]
        if x.dim() == 2:
            x_view = x.reshape(x.size(0), NUM_OPS, NUM_NODES, NUM_NODES)
        else:
            x_view = x

        # target: [batch, 4, 4]
        target_onehot = x_view.permute(0, 2, 3, 1)
        target = target_onehot.argmax(dim=-1)

        recon_logits, recon_probs, mu, logvar, acc_pred = model(x)

        # predizione operazione per ogni arco
        pred = recon_logits.argmax(dim=-1)

        correct += (pred == target).sum().item()
        total += target.numel()

    recon_acc = correct / total

    print(f"Reconstruction accuracy: {recon_acc * 100:.2f}%")

    return recon_acc
