import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
import os
import nasbench301 as nb
import random
#local
from dataset_loader import tensor_to_genotype, genotype_to_tensor, cell_to_tensor, tensor_to_cell
from model import vae_accuracy_loss,vae_accuracy_loss_nas301


OPS = {
    'nor_conv_3x3': 0,
    'nor_conv_1x1': 1,
    'skip_connect': 2,
    'avg_pool_3x3': 3,
    'none':         4,
    'zeroize':      5,
}
INV_OPS = {v: k for k, v in OPS.items()}

def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def build_accuracy_pairs(
    X,
    y,
    K=50,
    min_delta_acc=0.01,
    seed=42
):
    """
    Builds local improvement pairs for flow training.

    For each latent point x_i, the function searches among its K nearest
    neighbors for points x_j whose accuracy is higher by at least
    min_delta_acc. Among the improving neighbors, the one with the highest
    accuracy is selected.

    The target direction is:

        direction = x_j - x_i

    Input:
        X : Latent embeddings with shape [N, latent_dim].
        y : Accuracy values associated with X.
        K : Number of local nearest neighbors considered for each point.
        min_delta_acc : Minimum required accuracy improvement.

    Output:
        pairs_x : Starting latent points with shape [num_pairs, latent_dim].
        pairs_target : Target improvement directions with shape [num_pairs, latent_dim].
    """

    # converting to numpy array
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    y_np = y_np.reshape(-1)
    rng = np.random.default_rng(seed)

    K = min(len(X_np), K)

    nbrs = NearestNeighbors(
        n_neighbors=K
    ).fit(X_np)

    _, indices = nbrs.kneighbors(X_np)

    pairs_x = []
    pairs_target = []

    # list of improving neighbors for each X
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

        # scegli il vicino con accuracy più alta
        print("random")
        j = rng.choice(better)

        x_j = X_np[j]

        # target direction for flow matching
        direction = x_j - x_i

        pairs_x.append(x_i)
        pairs_target.append(direction)

    if len(pairs_x) == 0:
        pairs_x = torch.empty(
            (0, X_np.shape[1]),
            dtype=torch.float32
        )
        pairs_target = torch.empty(
            (0, X_np.shape[1]),
            dtype=torch.float32
        )
    else:
        pairs_x = torch.tensor(
            np.array(pairs_x),
            dtype=torch.float32
        )
        pairs_target = torch.tensor(
            np.array(pairs_target),
            dtype=torch.float32
        )

    print("Number of pairs:", len(pairs_x))

    return pairs_x, pairs_target


def generate_archs(dataset,N = 256):
    """Create initial dataset with N randomly generated architectures from dataset"""
  
    generator = torch.Generator().manual_seed(42)
    random_indices = torch.randperm(len(dataset), generator=generator)[:N]
    initial_dataset = Subset(dataset, random_indices.tolist())
    
    initial_loader = DataLoader(
        initial_dataset,
        batch_size=64,
        shuffle=True
    )

    return initial_loader

def decoded_x_to_nas201_arch(x_decoded):
    """
    Convert a decoded NAS201 vector into a NAS201 architecture string.

    The input is reshaped as [6, 4, 4], where A[:, src, dst] contains the operation
    scores for the edge src -> dst. For each valid edge, the operation with the
    maximum score is selected.
    """
    if not isinstance(x_decoded, torch.Tensor):
        x_decoded = torch.tensor(x_decoded, dtype=torch.float32)
    x_decoded = x_decoded.detach().cpu()

    A = x_decoded.view(6, 4, 4)
    nodes = []
    for dst in range(1, 4):
        edges = []
        for src in range(dst):
            op_idx = torch.argmax(A[:, src, dst]).item()
            op_name = INV_OPS[op_idx]

            if op_name == "zeroize":
                op_name = "none"
            #reconstructing the string
            edges.append(f"{op_name}~{src}")
        node_str = "|" + "|".join(edges) + "|"
        nodes.append(node_str)
    arch_str = "+".join(nodes)

    return arch_str


def train_one_epoch(
    flow,
    model_VAE,
    train_loader,
    beta=0.0,
    lambda_acc=1.0,
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

        vae_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        best_vae_loss = float("inf")
        patience_counter = 0

        for epoch in range(vae_epochs):

            model.train()

            total_vae_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_acc_loss = 0.0

            for x, y in train_loader:

                x = x.to(DEVICE).float()
                y = y.to(DEVICE).float().view(-1)

                recon_logits, recon_probs, mu, logvar, acc_pred = model(x)

                loss, recon_loss, kl, acc_loss = vae_accuracy_loss(
                    recon_logits=recon_logits,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    acc_pred=acc_pred,
                    true_acc=y,
                    beta=beta,
                    lambda_acc=lambda_acc
                )

                vae_optimizer.zero_grad()
                loss.backward()
                vae_optimizer.step()

                total_vae_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl.item()
                total_acc_loss += acc_loss.item()

            avg_vae_loss = total_vae_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            avg_acc_loss = total_acc_loss / len(train_loader)

            if epoch % 50 == 0:
                print(
                    f"VAE epoch {epoch:03d} | "
                    f"loss={avg_vae_loss:.6f} | "
                    f"recon={avg_recon_loss:.6f} | "
                    f"kl={avg_kl_loss:.6f} | "
                    f"acc_loss={avg_acc_loss:.6f}"
                )

            if early_stop:

                if avg_vae_loss < loss_threshold:
                    print(
                        f"Early stopping VAE: loss below threshold "
                        f"at epoch {epoch}, loss={avg_vae_loss:.6f}"
                    )
                    break

                if avg_vae_loss < best_vae_loss - min_delta:
                    best_vae_loss = avg_vae_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"Early stopping VAE: patience reached "
                        f"at epoch {epoch}, best_loss={best_vae_loss:.6f}"
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
    pairs_x, pairs_target = build_accuracy_pairs(
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

    pairs_dataset = TensorDataset(
        pairs_x,
        pairs_target
    )

    pairs_loader = DataLoader(
        pairs_dataset,
        batch_size=64,
        shuffle=True
    )

    # --------------------------------------------------
    # 4. Training flow matching
    # --------------------------------------------------
    flow_optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=1e-3
    )

    flow.train()

    for epoch in range(flow_epochs):

        total_flow_loss = 0.0

        for z_start, direction_target in pairs_loader:

            z_start = z_start.to(DEVICE).float()
            direction_target = direction_target.to(DEVICE).float()

            pred_direction = flow(z_start)

            loss = F.mse_loss(
                pred_direction,
                direction_target
            )

            flow_optimizer.zero_grad()
            loss.backward()
            flow_optimizer.step()

            total_flow_loss += loss.item()

        if epoch % 50 == 0:
            avg_flow_loss = total_flow_loss / len(pairs_loader)
            print(
                f"Flow epoch {epoch:03d} | "
                f"loss={avg_flow_loss:.6f}"
            )

    # --------------------------------------------------
    # 5. Generate new architectures from flow
    # --------------------------------------------------
    flow.eval()

    with torch.no_grad():

        z_start = z_all.to(DEVICE).float()

        direction = flow(z_start)

        z_new = z_start + alpha * direction

    print("z_new shape:", z_new.shape)

    return z_new, z_all, y_all

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

def pretrain_and_freeze_vae(
    model_VAE,
    pretrain_loader,
    loss_fn,                     # <--- nuova funzione passata in input
    beta=0.0,
    lambda_acc=1.0,
    vae_epochs=300,
    DEVICE="cpu",
    early_stop=True,
    patience=20,
    min_delta=1e-5,
    loss_threshold=1e-4,
    lr=1e-3,
    **loss_kwargs                # <--- eventuali parametri extra, tipo pos_weight_value
):
    """
    Preallena il VAE usando una loss_fn passata dall'esterno.
    Poi congela i parametri del VAE.
    """

    model_VAE = model_VAE.to(DEVICE)

    optimizer = torch.optim.Adam(
        model_VAE.parameters(),
        lr=lr
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(vae_epochs):

        model_VAE.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_acc_loss = 0.0

        for x, y in pretrain_loader:

            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)

            recon_logits, recon_probs, mu, logvar, acc_pred = model_VAE(x)

            loss, recon_loss, kl, acc_loss = loss_fn(
                recon_logits=recon_logits,
                recon_probs=recon_probs,
                x=x,
                mu=mu,
                logvar=logvar,
                acc_pred=acc_pred,
                true_acc=y,
                beta=beta,
                lambda_acc=lambda_acc,
                **loss_kwargs
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            total_acc_loss += acc_loss.item()

        avg_loss = total_loss / len(pretrain_loader)
        avg_recon = total_recon / len(pretrain_loader)
        avg_kl = total_kl / len(pretrain_loader)
        avg_acc_loss = total_acc_loss / len(pretrain_loader)

        if epoch % 50 == 0:
            print(
                f"VAE pretrain epoch {epoch:03d} | "
                f"loss={avg_loss:.6f} | "
                f"recon={avg_recon:.6f} | "
                f"kl={avg_kl:.6f} | "
                f"acc_loss={avg_acc_loss:.6f}"
            )

        if early_stop:

            if avg_loss < loss_threshold:
                print(
                    f"Early stopping: loss below threshold "
                    f"at epoch {epoch}, loss={avg_loss:.6f}"
                )
                break

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping: patience reached "
                    f"at epoch {epoch}, best_loss={best_loss:.6f}"
                )
                break

    model_VAE.eval()

    for p in model_VAE.parameters():
        p.requires_grad = False

    print("VAE pretrained and frozen.")

    return model_VAE
    
def load_nas301_performance_model():
    model_dir = os.path.join("nb_models_1.0", "xgb_v1.0")

    if not os.path.exists(model_dir):
        print("Scaricamento dei pesi dei modelli NAS-Bench-301...")
        nb.download_models(version="1.0")
    else:
        print("Pesi NAS-Bench-301 trovati localmente.")

    ensemble_dir_perf = os.path.join("nb_models_1.0", "xgb_v1.0")
    performance_model = nb.load_ensemble(ensemble_dir_perf)

    print("Surrogate model NAS-Bench-301 caricato con successo.")

    return performance_model

def decode_population_nas301(
    model_VAE,
    z_new,
    performance_model,
    DEVICE
            ):
    """
    Decodifica z_new in genotipi NAS301 e valuta con il surrogate.
    """

    model_VAE.eval()

    with torch.no_grad():

        decoded = model_VAE.decode(
            z_new.to(DEVICE).float()
        )

        # Se decode restituisce tuple, uso l'ultimo elemento.
        # Esempio: (recon_logits, recon_probs)
        if isinstance(decoded, tuple):
            x_new = decoded[-1]
        else:
            x_new = decoded

    x_new = x_new.detach().cpu()

    new_genotypes = []
    new_accs = []
    new_infos = []

    for i in range(x_new.shape[0]):

        x_decoded = x_new[i].view(-1)

        genotype = tensor_to_genotype(x_decoded)

        acc, info = query_nas301_accuracy(
            performance_model=performance_model,
            arch=genotype
        )

        if acc is None:
            continue

        new_genotypes.append(genotype)
        new_accs.append(acc)
        new_infos.append(info)

    return new_genotypes, new_accs, new_infos
def build_next_population_nas301(
    new_genotypes,
    new_accs,
    train_loader,
    elite_fraction=0.1,
    max_population_size=None
):
    """
    Nuova popolazione NAS301 a dimensione fissa:

        top flow-generated + top elite dalla popolazione corrente

    Se max_population_size è None, mantiene il comportamento espansivo.
    Se max_population_size è un intero, la popolazione finale avrà al massimo
    quella dimensione.
    """

    # --------------------------------------------------
    # 1. Architetture generate dal flow
    # --------------------------------------------------

    generated_rows = []

    for genotype, acc in zip(new_genotypes, new_accs):

        generated_rows.append({
            "arch": genotype,
            "arch_key": str(genotype),
            "acc": float(acc),
            "source": "flow"
        })

    generated_df = pd.DataFrame(generated_rows)

    generated_df = (
        generated_df
        .sort_values("acc", ascending=False)
        .drop_duplicates(subset=["arch_key"], keep="first")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 2. Popolazione corrente
    # --------------------------------------------------

    current_rows = []

    for batch_x, batch_y in train_loader:

        for x_curr, y_curr in zip(batch_x, batch_y):

            x_curr = x_curr.float().view(-1)
            acc_curr = float(y_curr)

            genotype_curr = tensor_to_genotype(x_curr)

            current_rows.append({
                "arch": genotype_curr,
                "arch_key": str(genotype_curr),
                "acc": acc_curr,
                "source": "elite"
            })

    df_current_population = pd.DataFrame(current_rows)

    df_current_population = (
        df_current_population
        .sort_values("acc", ascending=False)
        .drop_duplicates(subset=["arch_key"], keep="first")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. Calcolo quante elite e quante flow tenere
    # --------------------------------------------------

    if max_population_size is None:
        # comportamento vecchio: flow + elite, quindi può crescere
        n_elite = int(len(df_current_population) * elite_fraction)
        n_elite = max(0, n_elite)
        n_flow = len(generated_df)
    else:
        # comportamento nuovo: popolazione finale fissa
        n_elite = int(max_population_size * elite_fraction)
        n_elite = max(0, n_elite)

        n_flow = max_population_size - n_elite
        n_flow = max(1, n_flow)

    elite_df = df_current_population.head(n_elite).copy()
    flow_df = generated_df.head(n_flow).copy()

    # --------------------------------------------------
    # 4. Concat flow selezionate + elite selezionata
    # --------------------------------------------------

    df_next_population = pd.concat(
        [flow_df, elite_df],
        ignore_index=True
    )

    before_drop = len(df_next_population)

    # --------------------------------------------------
    # 5. Drop duplicati
    # --------------------------------------------------

    df_next_population = (
        df_next_population
        .sort_values("acc", ascending=False)
        .drop_duplicates(subset=["arch_key"], keep="first")
        .reset_index(drop=True)
    )

    after_drop = len(df_next_population)

    # Se dopo il drop duplicati siamo sotto max_population_size,
    # provo a riempire con altre flow generate non già presenti.
    if max_population_size is not None and len(df_next_population) < max_population_size:

        missing = max_population_size - len(df_next_population)
        existing_keys = set(df_next_population["arch_key"])

        extra_flow = generated_df[
            ~generated_df["arch_key"].isin(existing_keys)
        ].head(missing)

        if len(extra_flow) > 0:
            df_next_population = pd.concat(
                [df_next_population, extra_flow],
                ignore_index=True
            )

            df_next_population = (
                df_next_population
                .sort_values("acc", ascending=False)
                .drop_duplicates(subset=["arch_key"], keep="first")
                .reset_index(drop=True)
            )

    # Se ancora è più grande, tronco
    if max_population_size is not None:
        df_next_population = (
            df_next_population
            .head(max_population_size)
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # 6. Conversione in tensori
    # --------------------------------------------------

    X_next = []
    y_next = []

    for _, row in df_next_population.iterrows():

        genotype = row["arch"]
        acc = float(row["acc"])

        x = genotype_to_tensor(genotype)
        x = torch.from_numpy(x).float().view(-1)

        X_next.append(x)
        y_next.append(acc)

    X_next = torch.stack(X_next)
    y_next = torch.tensor(y_next).float().view(-1)

    print("\nNext population NAS301:")
    print(f"generated unique flow archs = {len(generated_df)}")
    print(f"selected flow archs         = {len(flow_df)}")
    print(f"selected elite archs        = {len(elite_df)}")
    print(f"before duplicate removal    = {before_drop}")
    print(f"after duplicate removal     = {after_drop}")
    print(f"final population size       = {len(df_next_population)}")
    print(f"mean acc                    = {df_next_population['acc'].mean():.4f}")
    print(f"max acc                     = {df_next_population['acc'].max():.4f}")

    return X_next, y_next, df_next_population

def query_nas301_accuracy(performance_model,arch,metric="val_accuracy"):
    """
    Query the surrogate model for architecture accuracy.
    """

    pred = performance_model.predict(
        config=arch,
        representation="genotype",
        with_noise=False
    )

    acc = float(pred)

    # NAS301 di solito restituisce percentuale, es. 93.42
    if acc > 1.5:
        acc = acc / 100.0

    info = {
        "raw_prediction": float(pred),
        "normalized_accuracy": acc,
        "metric": metric
    }

    return acc, info

def query_nas201_accuracy(
    api,
    arch_str,
    dataset_name="cifar10",
    hp="200",
    metric="test-accuracy"
):
    """
    Query the NAS-Bench-201/NATS-Bench API for the accuracy of a given architecture.

    Input: 
    dataset_name : Dataset used for evaluation. Valid values include:
        "cifar10", "cifar10-valid", "cifar100", "ImageNet16-120".

    hp : Training budget used by the benchmark. Valid values are "12" and "200".

    metric : Metric to retrieve from the benchmark information.
    """

    # find architecture Api index
    arch_index = api.query_index_by_arch(arch_str)
    if arch_index is None or arch_index < 0:
        return None, None
    #query performance info
    info = api.get_more_info(
        arch_index,
        dataset_name,
        hp=hp,
        is_random=False
    )
    acc = info.get(metric, None)

    return acc, info