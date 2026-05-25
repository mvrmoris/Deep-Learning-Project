import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from model import vae_loss,vae_accuracy_loss
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm


def build_accuracy_pairs(
    X,
    y,
    K=50,
    min_delta_acc=0.01,
    seed=42,
    return_info=True
):
    """
    Costruisce coppie locali nello spazio latente.

    Per ogni punto x_i, cerca tra i suoi K vicini un punto x_j
    con accuracy maggiore di almeno min_delta_acc.

    La target diventa la direzione:
        direction = x_j - x_i

    Parameters
    ----------
    X : np.ndarray oppure torch.Tensor
        Matrice degli embedding latenti, shape [N, latent_dim].

    y : np.ndarray oppure torch.Tensor
        Accuracy associate agli embedding, shape [N] oppure [N, 1].

    K : int
        Numero di vicini locali da considerare.

    min_delta_acc : float
        Miglioramento minimo richiesto su y.

    seed : int
        Seed per scegliere casualmente tra i vicini migliori.

    return_info : bool
        Se True, restituisce anche una lista con info sulle coppie.

    Returns
    -------
    pairs_x : torch.Tensor
        Punti di partenza, shape [num_pairs, latent_dim].

    pairs_target : torch.Tensor
        Direzioni verso punti migliori, shape [num_pairs, latent_dim].

    pairs_info : list[dict]
        Informazioni sulle coppie create.
    """

    # conversione robusta a numpy
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    # rende y vettore 1D
    y_np = y_np.reshape(-1)

    assert len(X_np) == len(y_np), "X e y devono avere la stessa lunghezza"

    # K non può superare il numero di punti
    K = min(K, len(X_np))

    rng = np.random.default_rng(seed)

    nbrs = NearestNeighbors(
        n_neighbors=K
    ).fit(X_np)

    distances, indices = nbrs.kneighbors(X_np)

    pairs_x = []
    pairs_target = []
    pairs_info = []

    for i in range(len(X_np)):

        x_i = X_np[i]
        acc_i = y_np[i]

        neigh_idx = indices[i]

        better = []

        for j in neigh_idx:

            if j == i:
                continue

            acc_j = y_np[j]

            if acc_j > acc_i + min_delta_acc:
                better.append(j)

        if len(better) == 0:
            continue

        j = rng.choice(better)

        x_j = X_np[j]
        acc_j = y_np[j]

        direction = x_j - x_i

        pairs_x.append(x_i)
        pairs_target.append(direction)

        if return_info:
            pairs_info.append({
                "i": int(i),
                "j": int(j),
                "acc_i": float(acc_i),
                "acc_j": float(acc_j),
                "delta_acc": float(acc_j - acc_i),
                "distance": float(np.linalg.norm(x_j - x_i))
            })

    if len(pairs_x) == 0:
        pairs_x = torch.empty((0, X_np.shape[1]), dtype=torch.float32)
        pairs_target = torch.empty((0, X_np.shape[1]), dtype=torch.float32)
    else:
        pairs_x = torch.tensor(np.array(pairs_x), dtype=torch.float32)
        pairs_target = torch.tensor(np.array(pairs_target), dtype=torch.float32)

    if return_info:
        return pairs_x, pairs_target, pairs_info

    return pairs_x, pairs_target


def generate_archs(dataset,N = 256):
    """generating architectures from actual distribution (for testing)"""
  
    generator = torch.Generator().manual_seed(42)

    random_indices = torch.randperm(len(dataset), generator=generator)[:N]

    # Crea il dataset iniziale con solo quelle N architetture
    initial_dataset = Subset(dataset, random_indices.tolist())
    
    # DataLoader iniziale
    initial_loader = DataLoader(
        initial_dataset,
        batch_size=64,
        shuffle=True
    )


    return initial_loader


OPS = {
    'nor_conv_3x3': 0,
    'nor_conv_1x1': 1,
    'skip_connect': 2,
    'avg_pool_3x3': 3,
    'none':         4,
    'zeroize':      5,
}
INV_OPS = {v: k for k, v in OPS.items()}

def arch_to_tensor(arch_str):
    """
    Converte una stringa NAS-Bench-201 / NATS-Bench TSS
    in un tensore A[op, src, dst] di shape (6, 4, 4).

    Esempio arch_str:
    '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+...'
    """

    nodes = arch_str.strip("|").split("+")
    N = len(nodes) + 1
    K = len(OPS)

    A = np.zeros((K, N, N), dtype=np.float32)

    pattern = r"(" + "|".join(OPS.keys()) + r")~(\d)"

    for dst, node in enumerate(nodes, start=1):
        edges = re.findall(pattern, node)

        for op_name, src_node in edges:
            src = int(src_node)
            A[OPS[op_name], src, dst] = 1.0

    return A

def decoded_x_to_nas201_arch(x_decoded):
    """
    Converte un vettore decoded di shape [96] oppure una matrice [6, 4, 4]
    in stringa NAS201/NAS-Bench-201.

    La matrice è interpretata come:
        A[op, src, dst]

    Per ogni arco valido src -> dst prende l'operazione con argmax.
    """

    if isinstance(x_decoded, torch.Tensor):
        x_decoded = x_decoded.detach().cpu()

    # da [96] a [6, 4, 4]
    if x_decoded.ndim == 1:
        A = x_decoded.view(6, 4, 4)
    elif x_decoded.ndim == 3:
        A = x_decoded
    else:
        raise ValueError(f"Shape non valida: {x_decoded.shape}")

    nodes = []

    # NAS201 ha 4 nodi: 0 input, 1, 2, 3 output
    # per ogni dst, considero tutti i src precedenti
    for dst in range(1, 4):

        edges = []

        for src in range(dst):

            op_idx = torch.argmax(A[:, src, dst]).item()
            op_name = INV_OPS[op_idx]

            # NAS201 standard non conosce "zeroize".
            # Se compare, lo tratto come "none".
            if op_name == "zeroize":
                op_name = "none"

            edges.append(f"{op_name}~{src}")

        node_str = "|" + "|".join(edges) + "|"
        nodes.append(node_str)

    arch_str = "+".join(nodes)

    return arch_str

def query_nas201_accuracy(
    api,
    arch_str,
    dataset_name="cifar10",
    hp="200",
    metric="test-accuracy"
):
    """
    Cerca una architettura NAS201 nella API e restituisce accuracy.

    dataset_name può essere:
        "cifar10"
        "cifar10-valid"
        "cifar100"
        "ImageNet16-120"

    hp può essere:
        "12"
        "200"
    """

    # trova indice architettura nella API
    arch_index = api.query_index_by_arch(arch_str)

    if arch_index is None or arch_index < 0:
        return None, None

    # recupera info prestazioni
    info = api.get_more_info(
        arch_index,
        dataset_name,
        hp=hp,
        is_random=False
    )

    acc = info.get(metric, None)

    return acc, info


def train_one_epoch(
    flow,
    model_VAE,
    train_loader,
    beta=0,
    vae_epochs=200,
    flow_epochs=100,
    alpha=0.5,
    DEVICE="cpu",
    train_vae=True,
    early_stop=True,
    patience=15,
    min_delta=1e-5,
    loss_threshold=1e-4
):
    """
    1. Eventualmente allena il VAE
    2. Estrae gli embedding z
    3. Costruisce coppie di miglioramento
    4. Allena il flow
    5. Genera nuovi z tramite flow
    """

    model = model_VAE.to(DEVICE)
    flow = flow.to(DEVICE)

    # --------------------------------------------------
    # 1. Training VAE, opzionale
    # --------------------------------------------------
    if train_vae and vae_epochs > 0:

        vae_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_vae_loss = float("inf")
        patience_counter = 0

        for epoch in range(vae_epochs):

            model.train()
            total_vae_loss = 0.0

            for x, y in train_loader:

                x = x.to(DEVICE).float()

                recon, mu, logvar, pred_acc = model(x)

                loss = vae_loss(
                    recon=recon,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    beta=beta
                )

                vae_optimizer.zero_grad()
                loss.backward()
                vae_optimizer.step()

                total_vae_loss += loss.item()

            avg_vae_loss = total_vae_loss / len(train_loader)

            if epoch % 10 == 0:
                print(f"VAE epoch {epoch:03d} | loss={avg_vae_loss:.6f}")

            if early_stop:

                if avg_vae_loss < loss_threshold:
                    print(
                        f"Early stopping VAE at epoch {epoch}: "
                        f"loss={avg_vae_loss:.6f} < threshold={loss_threshold}"
                    )
                    break

                if avg_vae_loss < best_vae_loss - min_delta:
                    best_vae_loss = avg_vae_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"Early stopping VAE at epoch {epoch}: "
                        f"no improvement for {patience} epochs. "
                        f"best_loss={best_vae_loss:.6f}"
                    )
                    break

    else:
        print("Skipping VAE training: using frozen/pretrained VAE.")
        model.eval()

    # --------------------------------------------------
    # 2. Embeddings extraction
    # --------------------------------------------------
    model.eval()

    z_all = []
    y_all = []

    with torch.no_grad():
        for x, y in train_loader:

            x = x.to(DEVICE).float()
            y = y.float().view(-1)

            mu, logvar = model.encode(x)

            z_all.append(mu.cpu())
            y_all.append(y.cpu())

    z_all = torch.cat(z_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    print("z_all shape:", z_all.shape)
    print("y_all shape:", y_all.shape)

    # --------------------------------------------------
    # 3. Pair generation
    # --------------------------------------------------
    pairs_x, pairs_target, pairs_info = build_accuracy_pairs(
        X=z_all,
        y=y_all,
        K=50,
        min_delta_acc=0.0,
        seed=42
    )

    if len(pairs_x) == 0:
        print("Nessuna coppia trovata: prova ad aumentare K o abbassare min_delta_acc.")
        return None

    print("pairs_x shape:", pairs_x.shape)
    print("pairs_target shape:", pairs_target.shape)

    pairs_dataset = TensorDataset(pairs_x, pairs_target)

    pairs_loader = DataLoader(
        pairs_dataset,
        batch_size=64,
        shuffle=True
    )

    # --------------------------------------------------
    # 4. Training flow matching
    # --------------------------------------------------
    flow_optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    flow.train()

    for epoch in range(flow_epochs):

        total_flow_loss = 0.0

        for z_start, direction_target in pairs_loader:

            z_start = z_start.to(DEVICE).float()
            direction_target = direction_target.to(DEVICE).float()

            pred_direction = flow(z_start)

            loss = F.mse_loss(pred_direction, direction_target)

            flow_optimizer.zero_grad()
            loss.backward()
            flow_optimizer.step()

            total_flow_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Flow epoch {epoch:03d} | loss={total_flow_loss:.6f}")

    # --------------------------------------------------
    # 5. Generate new architectures from flow
    # --------------------------------------------------
    flow.eval()

    with torch.no_grad():

        z_start = z_all.to(DEVICE).float()

        direction = flow(z_start)

        z_new = z_start + alpha * direction

    print("z_new shape:", z_new.shape)

    return z_new, z_all, y_all, pairs_info


def compare_accuracy_distributions(y_all, new_accs, title="NAS201 Accuracy: Initial vs Generated",path = None):

    # --- conversione dati ---
    if isinstance(y_all, torch.Tensor):
        y_init = y_all.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        y_init = np.array(y_all, dtype=np.float32).reshape(-1)

    y_gen = np.array([a for a in new_accs if a is not None], dtype=np.float32)

    if len(y_gen) == 0:
        raise ValueError("new_accs contiene solo valori None.")

    # porta tutto in percentuale se normalizzato 0-1
    if y_init.max() <= 1.5:
        y_init = y_init * 100.0
    if y_gen.max() <= 1.5:
        y_gen = y_gen * 100.0

    # --- calcolo mean e varianza ---
    mu_i,  std_i,  var_i  = y_init.mean(), y_init.std(),  y_init.var()
    mu_g,  std_g,  var_g  = y_gen.mean(),  y_gen.std(),   y_gen.var()

    print(f"INITIAL    — n={len(y_init):4d}  mean={mu_i:.3f}  var={var_i:.3f}  std={std_i:.3f}  min={y_init.min():.3f}  max={y_init.max():.3f}")
    print(f"GENERATED  — n={len(y_gen):4d}  mean={mu_g:.3f}  var={var_g:.3f}  std={std_g:.3f}  min={y_gen.min():.3f}  max={y_gen.max():.3f}")
    print(f"Δ mean = {mu_g - mu_i:+.3f}")

    # --- asse x ---
    margin = max(std_i, std_g) * 4.0
    xs     = np.linspace(min(mu_i, mu_g) - margin, max(mu_i, mu_g) + margin, 800)
    pdf_i  = norm.pdf(xs, mu_i, std_i)
    pdf_g  = norm.pdf(xs, mu_g, std_g)

    # --- colori e stile ---
    C_INIT, C_GEN, BG = "#4A90D9", "#C0503A", "#F7F7F5"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#CCCCCC")
    ax.spines["bottom"].set_edgecolor("#CCCCCC")

    # fill e curve
    ax.fill_between(xs, pdf_i, alpha=0.18, color=C_INIT)
    ax.fill_between(xs, pdf_g, alpha=0.18, color=C_GEN)
    ax.plot(xs, pdf_i, color=C_INIT, linewidth=2.2,
            label=f"Initial   (μ={mu_i:.2f},  σ²={var_i:.2f})")
    ax.plot(xs, pdf_g, color=C_GEN,  linewidth=2.2,
            label=f"Generated (μ={mu_g:.2f},  σ²={var_g:.2f})")

    # medie tratteggiate
    ax.axvline(mu_i, color=C_INIT, linewidth=1.3, linestyle="--", alpha=0.9)
    ax.axvline(mu_g, color=C_GEN,  linewidth=1.3, linestyle="--", alpha=0.9)

    # griglia e assi
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Accuracy (%)", fontsize=11, labelpad=8, color="#444444")
    ax.set_ylabel("Density",      fontsize=11, labelpad=8, color="#444444")
    ax.tick_params(colors="#666666", labelsize=9.5)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0, top=max(pdf_i.max(), pdf_g.max()) * 1.18)

    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#CCCCCC",
              loc="upper left", handlelength=1.8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if path is not None:
        plt.savefig(path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()

def train_vae_only(
    model_VAE,
    train_loader,
    beta=0,
    vae_epochs=200,
    lr=1e-3,
    DEVICE="cpu",
    early_stop=True,
    patience=15,
    min_delta=1e-5,
    loss_threshold=1e-4
):
    model = model_VAE.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(vae_epochs):

        model.train()
        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(DEVICE).float()

            recon, mu, logvar, pred_acc = model(x)

            loss = vae_loss(
                recon=recon,
                x=x,
                mu=mu,
                logvar=logvar,
                beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"VAE pretrain epoch {epoch:03d} | loss={avg_loss:.6f}")

        if early_stop:

            if avg_loss < loss_threshold:
                print(
                    f"Early stopping VAE at epoch {epoch}: "
                    f"loss={avg_loss:.6f} < threshold={loss_threshold}"
                )
                break

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping VAE at epoch {epoch}: "
                    f"no improvement for {patience} epochs. "
                    f"best_loss={best_loss:.6f}"
                )
                break

    return model


def pretrain_and_freeze_vae(
    model_VAE,
    pretrain_loader,
    beta=0,
    vae_epochs=300,
    DEVICE="cpu",
    early_stop=True,
    patience=20,
    min_delta=1e-5,
    loss_threshold=1e-4,
    lr=1e-3
):
    """
    Preallena il VAE su un loader grande e poi congela i suoi parametri.
    Usa solo la VAE loss: reconstruction + KL.
    """

    model_VAE = model_VAE.to(DEVICE)
    optimizer = torch.optim.Adam(model_VAE.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(vae_epochs):

        model_VAE.train()
        total_loss = 0.0

        for x, y in pretrain_loader:

            x = x.to(DEVICE).float()

            recon, mu, logvar, pred_acc = model_VAE(x)

            loss = vae_loss(
                recon=recon,
                x=x,
                mu=mu,
                logvar=logvar,
                beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(pretrain_loader)

        if epoch % 10 == 0:
            print(f"VAE pretrain epoch {epoch:03d} | loss={avg_loss:.6f}")

        if early_stop:

            if avg_loss < loss_threshold:
                print(
                    f"Early stopping VAE pretrain at epoch {epoch}: "
                    f"loss={avg_loss:.6f} < threshold={loss_threshold}"
                )
                break

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping VAE pretrain at epoch {epoch}: "
                    f"no improvement for {patience} epochs. "
                    f"best_loss={best_loss:.6f}"
                )
                break

    # --------------------------------------------------
    # Freeze VAE
    # --------------------------------------------------
    model_VAE.eval()

    for p in model_VAE.parameters():
        p.requires_grad = False

    print("VAE pretrained and frozen.")

    return model_VAE