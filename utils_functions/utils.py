import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset, DataLoader,TensorDataset
import os
import nasbench301 as nb
import random
import tempfile
from dataset_loader import tensor_to_genotype, genotype_to_tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def set_seed(seed=42, deterministic=True):
    #support function to set the seed 
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_numpy(x):
    #support function to convert tensor to numpy array 
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def build_accuracy_pairs(
    X,
    y,
    K=50,
    min_delta_acc=0.01,
    seed=42
):
    """
    For each latent point x_i, the function searches among its K nearest
    neighbors for points x_j whose accuracy is higher by at least
    min_delta_acc. Among the improving neighbors, the one with the highest
    accuracy is selected.
    """
    # converting to numpy array
    X_np = to_numpy(X)
    y_np = to_numpy(y)
    y_np = y_np.reshape(-1)
    rng = np.random.default_rng(seed)

    #computing indxs of neighbors
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

        #target direction for flow matching as x-j - x-i
        direction = x_j - x_i
        pairs_x.append(x_i)
        pairs_target.append(direction)

    #returning tensors
    pairs_x = torch.tensor(np.array(pairs_x),dtype=torch.float32)
    pairs_target = torch.tensor(np.array(pairs_target),dtype=torch.float32)

    print("Number of pairs:", len(pairs_x))
    return pairs_x, pairs_target

def generate_archs(data, converter, N=256, seed=42):
    """Seleziona N elementi casuali e li converte in architetture."""
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(data),
        generator=generator
    )[:min(N, len(data))].tolist()

    architectures = []

    for i in indices:
        item = data[i]

        # Gestisce Tensor, TensorDataset e dataset (x, y)
        x = item[0] if isinstance(item, (tuple, list)) else item

        architectures.append(
            converter(x.float().flatten())
        )

    return architectures

def random_nas201_arch():
    operations = [
        "nor_conv_3x3",
        "nor_conv_1x1",
        "skip_connect",
        "avg_pool_3x3",
        "none"
    ]

    nodes = []

    for dst in range(1, 4):
        edges = []

        for src in range(dst):
            op = random.choice(operations)
            edges.append(f"{op}~{src}")

        nodes.append(
            "|" + "|".join(edges) + "|"
        )

    return "+".join(nodes)

def load_nas301_performance_model():
    """Return the performance model of NAS301"""
    model_dir = os.path.join("nb_models_1.0", "xgb_v1.0")

    if os.path.exists(model_dir):
        print("Pesi NAS-Bench-301 trovati localmente.")
    else:
        print("Scaricamento dei pesi NAS-Bench-301...")
        nb.download_models(version="1.0")

    model = nb.load_ensemble(model_dir)
    print("Surrogate model NAS-Bench-301 caricato con successo.")
    return model

def decode_population_nas301(model_VAE, z_new, performance_model, DEVICE):
    """decode latent vectors and query surrogate model"""

    model_VAE.eval()
    #decode architectures 
    with torch.no_grad():
        x_new = model_VAE.decode(z_new.to(DEVICE).float())[-1].cpu()

    genotypes, accs, infos = [], [], []

    #convert into NAS301 valid genotypes and query surrogate model for accuracy
    for x in x_new:
        genotype = tensor_to_genotype(x.flatten())
        acc, info = query_nas301_accuracy(performance_model, genotype)
        if acc is not None:
            genotypes.append(genotype)
            accs.append(acc)
            infos.append(info)

    return genotypes, accs, infos

def build_next_population(
    decoded_archs,
    current_pop,
    current_accs,
    converter,
    elite_fraction=0.1,
    max_population_size=256,
    key_fn=str,
):
    """
    Costruisce la popolazione successiva unendo:
    - architetture generate dal flow;
    - migliori architetture della popolazione corrente.
    """

    # Conversione dei tensori decodificati in architetture
    generated_archs = [
        converter(x)
        for x in decoded_archs
    ]

    generated_df = pd.DataFrame({
        "arch": generated_archs,
        "source": "flow",
    })

    generated_df["arch_key"] = generated_df["arch"].map(key_fn)

    generated_df = (
        generated_df
        .drop_duplicates("arch_key")
        .reset_index(drop=True)
    )

    # Popolazione corrente, già valutata
    current_df = pd.DataFrame({
        "arch": current_pop,
        "acc": list(map(float, current_accs)),
        "source": "current",
    })

    current_df["arch_key"] = current_df["arch"].map(key_fn)

    current_df = (
        current_df
        .sort_values("acc", ascending=False)
        .drop_duplicates("arch_key")
        .reset_index(drop=True)
    )

    n_elite = int(max_population_size * elite_fraction)
    n_generated = max_population_size - n_elite

    elite_df = current_df.head(n_elite).copy()
    elite_df["source"] = "elite"

    generated_df = generated_df.head(n_generated)

    # Le nuove architetture non hanno ancora accuracy
    elite_df = elite_df.drop(columns="acc")

    next_df = (
        pd.concat(
            [generated_df, elite_df],
            ignore_index=True
        )
        .drop_duplicates("arch_key")
        .head(max_population_size)
        .reset_index(drop=True)
    )

    return next_df["arch"].tolist(), next_df

def query_nas301_accuracy(
    performance_model,
    arch,
    metric="val_accuracy"):

    raw_pred = float(performance_model.predict(
        config=arch,
        representation="genotype",
        with_noise=False,
    ))

    acc = raw_pred / 100.0 if raw_pred > 1.5 else raw_pred

    return acc, {
        "raw_prediction": raw_pred,
        "normalized_accuracy": acc,
        "metric": metric,
    }

def query_nas201_accuracy(
    api,
    arch_str,
    dataset_name="cifar10",
    hp="200",
    metric="test-accuracy"
):
    """Query the NAS-Bench-201/NATS-Bench API for the accuracy of a given architecture."""

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

#NAS201 operations
OPS = {
    'nor_conv_3x3': 0,
    'nor_conv_1x1': 1,
    'skip_connect': 2,
    'avg_pool_3x3': 3,
    'none':         4
}
INV_OPS = {v: k for k, v in OPS.items()}

def decoded_x_to_nas201_arch(x_decoded):
    """
    Convert a decoded NAS201 vector into a NAS201 architecture string.

    The input is reshaped as [5,4, 4], where A[:, src, dst] contains the operation
    scores for the edge src -> dst. For each valid edge, the operation with the
    maximum score is selected.
    """
    A = to_numpy(x_decoded).reshape(5, 4, 4)

    nodes = []

    for dst in range(1, 4):
        edges = []

        for src in range(dst):
            op_idx = np.argmax(A[:, src, dst])
            edges.append(f"{INV_OPS[op_idx]}~{src}")

        nodes.append(f"|{'|'.join(edges)}|")

    return "+".join(nodes)

def get_cifar10_loaders(batch_size=256, num_workers=2):
    """load dataset to train models"""
    tmp_dir = os.path.join(tempfile.gettempdir(), 'cifar10_data')

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
        root=tmp_dir, train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=tmp_dir, train=False, download=True, transform=transform_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def load_csv_as_dataset(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith("x_")]

    feature_cols = sorted(
        feature_cols,
        key=lambda c: int(c.split("_")[1])
    )

    X = df[feature_cols].values
    Y = df["accuracy"].values

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X, Y)

    print("CSV caricato:", csv_path)
    print("Numero esempi:", len(dataset))
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Accuracy min:", Y.min().item())
    print("Accuracy max:", Y.max().item())

    return X, Y, dataset

