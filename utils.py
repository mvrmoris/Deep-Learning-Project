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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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

        #choosing random neighbor
        j = rng.choice(better)

        x_j = X_np[j]

        #target direction for flow matching
        direction = x_j - x_i

        pairs_x.append(x_i)
        pairs_target.append(direction)

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

def load_nas301_performance_model():
    "returns NAS301 performance model"
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

def decode_population_nas301(model_VAE, z_new, performance_model, DEVICE):
    """
    Decode z_new genotypes and queries the surrogate model for accuracy
    """

    model_VAE.eval()
    with torch.no_grad():

        decoded = model_VAE.decode(
            z_new.to(DEVICE).float()
        )
        x_new = decoded[-1]
    x_new = x_new.detach().cpu()

    new_genotypes = []
    new_accs = []
    new_infos = []

    for i in range(x_new.shape[0]):
        x_decoded = x_new[i].view(-1)
        genotype = tensor_to_genotype(x_decoded)
        acc, info = query_nas301_accuracy(
            performance_model=performance_model,
            arch=genotype)

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
    max_population_size=256
):
    """
    Builds the next NAS301 population with a fixed maximum size.
    The next population is composed of:
        top flow-generated architectures + top elite architectures
        from the current population.

    The number of elite architectures is determined by `elite_fraction`,
    while the remaining slots are filled with the best flow-generated
    architectures. Duplicates are removed using the string representation
    of the genotype as a unique key.
    """

    # 1. Architectures generated from flow
    generated_rows = []
    for genotype, acc in zip(new_genotypes, new_accs):
        generated_rows.append({
            "arch": genotype,
            "arch_key": str(genotype),
            "acc": float(acc),
            "source": "flow"
        })
    generated_df = pd.DataFrame(generated_rows)

    generated_df = (generated_df.sort_values("acc", ascending=False)
        .drop_duplicates(subset=["arch_key"], keep="first")
        .reset_index(drop=True)
    )
    #2. current population
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

    # 3. Compute how many elite and flow generated architectures to keep in the new population
    n_elite = int(max_population_size * elite_fraction)
    n_elite = max(0, n_elite)
    n_flow = max_population_size - n_elite
    n_flow = max(1, n_flow)
    elite_df = df_current_population.head(n_elite).copy()
    flow_df = generated_df.head(n_flow).copy()

    # 4. Concat flow and elite
    df_next_population = pd.concat(
        [flow_df, elite_df],
        ignore_index=True
    )
    before_drop = len(df_next_population)

    # 5. Drop duplicati
    df_next_population = (
        df_next_population
        .sort_values("acc", ascending=False)
        .drop_duplicates(subset=["arch_key"], keep="first")
        .reset_index(drop=True)
    )
    after_drop = len(df_next_population)

    # 6. Conversione in tensori
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

def get_cifar10_loaders(batch_size=256, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
