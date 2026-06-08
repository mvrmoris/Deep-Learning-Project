import os
import tarfile
import re
import numpy as np
import torch
from nats_bench import create
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import namedtuple
import random
import nasbench301 as nb


#NAS301 structure utils: 
PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

N_OPS = len(PRIMITIVES)       
N_NODES = 4                   
N_INPUTS = 2                 
TOTAL_NODES = N_INPUTS + N_NODES  #

OP_TO_IDX = {op: i for i, op in enumerate(PRIMITIVES)}

Genotype = namedtuple(
    "Genotype",
    "normal normal_concat reduce reduce_concat"
)

import os
import nasbench301 as nb
import xgboost as xgb


def load_nas301_performance_model(
    version="1.0",
    model_type="xgb_v1.0",
):
    """Load the NAS301 surrogate model, downloading it if necessary."""

    datasets_dir = os.path.dirname(os.path.abspath(__file__))

    model_dir = os.path.join(
        datasets_dir,
        f"nb_models_{version}",
        model_type,
    )

    if not os.path.exists(model_dir):
        print(
            "Pesi NAS-Bench-301 non trovati, "
            "scaricamento in corso..."
        )

        nb.download_models(
            version=version,
            download_dir=datasets_dir,
        )
    else:
        print(f"Pesi NAS-Bench-301 trovati in {model_dir}")

    with xgb.config_context(verbosity=0):
        model = nb.load_ensemble(model_dir)

    print("Surrogate NAS-Bench-301 caricato.")

    return model

def random_cell():
    """
    Generate a casual DARTS cell
    """
    edges = []

    for node_idx in range(N_NODES):

        n_available = N_INPUTS + node_idx
        srcs = random.sample(range(n_available), 2)

        for src in sorted(srcs):
            op = random.choice(PRIMITIVES)
            edges.append((op, src))

    return edges


def random_genotype():
    return Genotype(
        normal=random_cell(),
        normal_concat=list(range(N_INPUTS, TOTAL_NODES)),
        reduce=random_cell(),
        reduce_concat=list(range(N_INPUTS, TOTAL_NODES)),
    )

def cell_to_tensor(cell_edges):
    """Convert a DARTS cell to a one-hot tensor."""
    A = np.zeros(
        (N_OPS, TOTAL_NODES, TOTAL_NODES),
        dtype=np.float32,
    )

    for dst_offset in range(N_NODES):
        dst = N_INPUTS + dst_offset

        for op_name, src in cell_edges[
            2 * dst_offset : 2 * dst_offset + 2
        ]:
            A[OP_TO_IDX[op_name], src, dst] = 1.0

    return A

def genotype_to_tensor(genotype):
    """frm genotype to tensor"""

    normal_A = cell_to_tensor(genotype.normal)
    reduce_A = cell_to_tensor(genotype.reduce)

    both = np.stack([normal_A, reduce_A])

    return both.flatten()



def tensor_to_cell(x_cell):
    """Convert a tensor into a valid NAS301 cell."""
    if isinstance(x_cell, torch.Tensor):
        x_cell = x_cell.detach().cpu().numpy()

    edges = []

    for dst in range(N_INPUTS, TOTAL_NODES):
        candidates = []

        for src in range(dst):
            op_idx = int(np.argmax(x_cell[:, src, dst]))

            candidates.append((
                x_cell[op_idx, src, dst],
                PRIMITIVES[op_idx],
                src,
            ))

        selected = sorted(
            candidates,
            key=lambda x: x[0],
            reverse=True,
        )[:2]

        edges.extend(
            (op_name, src)
            for _, op_name, src in sorted(
                selected,
                key=lambda x: x[2],
            )
        )
    return edges

def tensor_to_genotype(x):
    """Convert a tensor into a NAS301 genotype."""
    x = torch.as_tensor(
        x,
        dtype=torch.float32
    ).detach().cpu().view(
        2,
        N_OPS,
        TOTAL_NODES,
        TOTAL_NODES
    )
    concat = list(range(N_INPUTS, TOTAL_NODES))

    return Genotype(
        normal=tensor_to_cell(x[0]),
        normal_concat=concat,
        reduce=tensor_to_cell(x[1]),
        reduce_concat=concat.copy(),
    )

class NAS301TorchDatasetBuilder:
    """
    Costruisce un dataset Torch per NAS-Bench-301 campionando genotipi DARTS
    casuali e stimando la performance con il surrogate model.

    X:
        tensore delle architetture, shape:
            (num_archs, 2, 7, 6, 6)

        oppure flattened:
            (num_archs, 504)

    y:
        accuracy stimata dal surrogate, normalizzata in [0, 1] se normalize_y=True.
    """

    def __init__(
        self,
        performance_model,
        n_samples=5000,
        flatten=True,
        normalize_y=True,
        seed=42,
        verbose=True,
    ):
        self.performance_model = performance_model
        self.n_samples = n_samples
        self.flatten = flatten
        self.normalize_y = normalize_y
        self.seed = seed
        self.verbose = verbose

        self.genotypes = None

    def _predict_accuracy(self, genotype):
        acc = self.performance_model.predict(
            config=genotype,
            representation="genotype",
            with_noise=False
        )

        return float(acc)

    def build_tensors(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        X_list = []
        y_list = []
        genotypes = []

        if self.verbose:
            print(
                f"Campionamento di {self.n_samples} architetture DARTS "
                "per NAS-Bench-301..."
            )

        for i in range(self.n_samples):

            genotype = random_genotype()
            x = genotype_to_tensor(genotype)

            if not self.flatten:
                x = x.reshape(
                    2,
                    N_OPS,
                    TOTAL_NODES,
                    TOTAL_NODES
                )

            acc = self._predict_accuracy(genotype)

            genotypes.append(genotype)
            X_list.append(x)
            y_list.append(acc)

            if self.verbose and i % 2000 == 0:
                print(
                    f"  [{i:>5}/{self.n_samples}] "
                    f"acc media finora: {np.mean(y_list):.2f}%"
                )

        X = torch.tensor(
            np.stack(X_list),
            dtype=torch.float32
        )

        y = torch.tensor(
            y_list,
            dtype=torch.float32
        )

        if self.normalize_y:
            y = y / 100.0

        self.genotypes = genotypes

        if self.verbose:
            y_np = np.array(y_list, dtype=np.float32)

            print("\nNAS301 sampled dataset:")
            print(f"X shape  : {tuple(X.shape)}")
            print(f"y shape  : {tuple(y.shape)}")
            print(
                f"Accuracy — min: {y_np.min():.2f}% | "
                f"max: {y_np.max():.2f}% | "
                f"mean: {y_np.mean():.2f}%"
            )

        return X, y

    def build_dataset(self):
        X, y = self.build_tensors()
        return TensorDataset(X, y)

    def build_loaders(
        self,
        train_ratio=0.8,
        batch_size=128,
        seed=42,
        shuffle_train=True,
    ):
        dataset = self.build_dataset()

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_dataset, test_dataset, train_loader, test_loader
