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
import pandas as pd
from ws_universale.nb201 import networkdag_to_nb201_str

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

#NAS201 structure utils
OPS = {
    "nor_conv_3x3": 0,
    "nor_conv_1x1": 1,
    "skip_connect": 2,
    "avg_pool_3x3": 3,
    "none": 4,
}


DATASET_NAME_MAP = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet16-120": "ImageNet16-120",
}

def load_nas301_performance_model(
    model_dir=None,
    version="1.0",
    model_type="xgb_v1.0"
):
    """
    Carica il performance model NAS301 dalla cartella padre.
    Se non è presente, scarica i pesi.
    """

    if model_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_dir = os.path.abspath(
            os.path.join(
                base_dir,
                "..",
                "nb_models_1.0",
                model_type
            )
        )

    if not os.path.exists(model_dir):
        print("Scaricamento dei pesi NAS-Bench-301 in corso...")
        nb.download_models(version=version)
    else:
        print("Pesi NAS-Bench-301 trovati localmente.")

    performance_model = nb.load_ensemble(model_dir)

    print("Surrogate model NAS-Bench-301 caricato con successo.")

    return performance_model

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
    """
    Convert DARTS cell into one-hot vector
    """
    A = np.zeros(
        (N_OPS, TOTAL_NODES, TOTAL_NODES),
        dtype=np.float32
    )

    edge_idx = 0

    for dst_offset in range(N_NODES):

        dst = N_INPUTS + dst_offset

        for _ in range(2):
            op_name, src = cell_edges[edge_idx]
            A[OP_TO_IDX[op_name], src, dst] = 1.0
            edge_idx += 1

    return A


def genotype_to_tensor(genotype):
    """
    frm genotype to tensor
    """
    normal_A = cell_to_tensor(genotype.normal)
    reduce_A = cell_to_tensor(genotype.reduce)

    both = np.stack([normal_A, reduce_A])

    return both.flatten()



def tensor_to_cell(x_cell):
    """
    Converte una cella NAS301 da tensore continuo a lista DARTS valida.

    Input:
        x_cell shape: (N_OPS, TOTAL_NODES, TOTAL_NODES)
        cioè (7, 6, 6)

    Per ogni nodo intermedio dst:
        1. per ogni source node src possibile, sceglie l'operazione migliore;
        2. tra i source node sceglie i due migliori, garantendo src distinti.

    Output:
        lista di 8 coppie (op_name, src), cioè 2 archi per ciascuno dei 4 nodi intermedi.
    """

    if isinstance(x_cell, torch.Tensor):
        x_cell = x_cell.detach().cpu().numpy()

    edges = []

    for dst_offset in range(N_NODES):

        dst = N_INPUTS + dst_offset

        src_candidates = []

        # Per ogni sorgente possibile scelgo la migliore operazione
        for src in range(dst):

            op_scores = x_cell[:, src, dst]

            op_idx = int(np.argmax(op_scores))
            op_name = PRIMITIVES[op_idx]
            score = float(op_scores[op_idx])

            src_candidates.append((score, op_name, src))

        # Ora scelgo i due source node migliori.
        # Sono distinti per costruzione, perché src_candidates contiene
        # una sola entry per ogni src.
        src_candidates = sorted(
            src_candidates,
            key=lambda t: t[0],
            reverse=True
        )

        selected = src_candidates[:2]

        # Ordine stabile per compatibilità DARTS/NAS301
        selected = sorted(
            selected,
            key=lambda t: t[2]
        )

        for _, op_name, src in selected:
            edges.append((op_name, src))

    return edges
def tensor_to_genotype(x):
    """
    Converte un vettore/tensore NAS301 in Genotype DARTS.

    Input:
        x shape: (504,)
        oppure (2, 7, 6, 6)
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()

    x = torch.tensor(x, dtype=torch.float32)

    x = x.view(
        2,
        N_OPS,
        TOTAL_NODES,
        TOTAL_NODES
    )

    normal_tensor = x[0]
    reduce_tensor = x[1]

    normal_cell = tensor_to_cell(normal_tensor)
    reduce_cell = tensor_to_cell(reduce_tensor)

    genotype = Genotype(
        normal=normal_cell,
        normal_concat=list(range(N_INPUTS, TOTAL_NODES)),
        reduce=reduce_cell,
        reduce_concat=list(range(N_INPUTS, TOTAL_NODES)),
    )

    return genotype

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

def load_nas201_api(
    datasets_dir=None,
    tar_name="NATS-tss-v1_0-3ffb9-simple.tar",
    extracted_name="NATS-tss-v1_0-3ffb9-simple",
    fast_mode=True,
    verbose=False
):
    """
    Carica automaticamente la API NAS-Bench-201 / NATS-Bench TSS.

    Se la cartella estratta non esiste, estrae il file .tar.
    """

    if datasets_dir is None:
        # File corrente: Deep-Learning-Project/dataset_loader.py
        # Dataset attesi in: Deep-Learning-Project/datasets
        base_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.abspath(
            os.path.join(base_dir, "datasets")
        )

    percorso_tar = os.path.join(datasets_dir, tar_name)
    dataset_path = os.path.join(datasets_dir, extracted_name)

    if not os.path.exists(dataset_path):
        if not os.path.exists(percorso_tar):
            raise FileNotFoundError(
                f"Archivio NAS201 non trovato:\n{percorso_tar}\n\n"
                "Controlla che il file .tar sia nella cartella datasets."
            )

        print("Estrazione NAS201 in corso...")

        with tarfile.open(percorso_tar, "r") as tar:
            tar.extractall(path=datasets_dir)

        print("Estrazione completata!")
    else:
        print("Dataset NAS201 già estratto.")

    api = create(
        dataset_path,
        "tss",
        fast_mode=fast_mode,
        verbose=verbose
    )

    print(f"Architetture NAS201 totali: {len(api)}")

    return api

class NAS201TorchDatasetBuilder:
    """
    Costruisce automaticamente X e y a partire dall'API NAS-Bench-201/NATS-Bench.

    X:
        tensore delle architetture, shape:
            (num_archs, 6, 4, 4)

        oppure flattened:
            (num_archs, 96)

    y:
        accuracy normalizzata in [0, 1], shape:
            (num_archs,)
    """

    def __init__(
        self,
        api,
        dataset_name="cifar10",
        metric="test-accuracy",
        hp="200",
        flatten=True,
        normalize_y=True,
    ):
        self.api = api
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        self.metric = metric
        self.hp = hp
        self.flatten = flatten
        self.normalize_y = normalize_y

    def _normalize_dataset_name(self, dataset_name):
        key = dataset_name.lower()

        if key not in DATASET_NAME_MAP:
            raise ValueError(
                f"Dataset non valido: {dataset_name}. "
                f"Scegli tra: {list(DATASET_NAME_MAP.keys())}"
            )

        return DATASET_NAME_MAP[key]

    def _get_arch_str(self, idx):
        """
        Recupera la stringa dell'architettura dall'API.

        Funziona con API che espongono:
            api.arch(idx)

        oppure:
            api.meta_archs[idx]

        Se la tua API usa un altro nome, basta modificare questo metodo.
        """

        if hasattr(self.api, "arch"):
            return self.api.arch(idx)

        if hasattr(self.api, "meta_archs"):
            return self.api.meta_archs[idx]

        if hasattr(self.api, "arch2infos_dict"):
            # fallback meno ideale, dipende dalla struttura interna
            return list(self.api.arch2infos_dict.keys())[idx]

        raise AttributeError(
            "Non riesco a trovare le architetture nell'API. "
            "Adatta il metodo _get_arch_str alla tua API."
        )

    def _get_accuracy(self, idx):
        info = self.api.get_more_info(
            idx,
            self.dataset_name,
            iepoch=None,
            hp=self.hp,
            is_random=False
        )

        acc = info.get(self.metric)

        if acc is None:
            raise ValueError(
                f"Metrica '{self.metric}' non trovata per idx={idx}, "
                f"dataset={self.dataset_name}, hp={self.hp}. "
                f"Chiavi disponibili: {list(info.keys())}"
            )

        return float(acc)

    def build_tensors(self):
        X_list = []
        y_list = []

        for idx in range(len(self.api)):
            arch_str = self._get_arch_str(idx)

            x = arch_to_tensor(arch_str)

            if self.flatten:
                x = x.reshape(-1)

            acc = self._get_accuracy(idx)

            X_list.append(x)
            y_list.append(acc)

            if hasattr(self.api, 'arch2infos_dict'):
                self.api.arch2infos_dict.pop(idx, None)
            if hasattr(self.api, 'arch2infos_full'):
                self.api.arch2infos_full.pop(idx, None)
            if hasattr(self.api, 'arch2infos_less'):
                self.api.arch2infos_less.pop(idx, None)

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)

        if self.normalize_y:
            y = y / 100.0

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

class NASDatasetFactory:

    @staticmethod
    def create(
        benchmark_name,
        *,
        api=None,
        performance_model=None,
        dataset_name="cifar10",
        metric="test-accuracy",
        hp="200",
        n_samples=5000,
        flatten=True,
        normalize_y=True,
        seed=42,
    ):
        benchmark_name = benchmark_name.upper()

        if benchmark_name == "NAS201":

            if api is None:
                api = load_nas201_api()

            builder = NAS201TorchDatasetBuilder(
                api=api,
                dataset_name=dataset_name,
                metric=metric,
                hp=hp,
                flatten=flatten,
                normalize_y=normalize_y,
        )

            return  builder.build_loaders()

        elif benchmark_name == "NAS301":

            if performance_model is None:
                performance_model = load_nas301_performance_model()

            builder = NAS301TorchDatasetBuilder(
                performance_model=performance_model,
                n_samples=n_samples,
                flatten=flatten,
                normalize_y=normalize_y,
                seed=seed,
            )
            train_dataset,test_dataset,train_loader,test_loader= builder.build_loaders()
            return train_dataset, test_dataset,train_loader, test_loader,performance_model

        else:
            raise ValueError(
                f"Benchmark non supportato: {benchmark_name}. "
                "Scegli tra NAS201 e NAS301."
            )
        
def fetch_gt_accuracies(
    networks   : list,           # list[NetworkDAG]
    accuracies : list[float],    # proxy accuracies dalla supernet
    dataset    : str = 'cifar10-valid',
    hp         : str = '200',
) -> pd.DataFrame:
    """
    Per ogni NetworkDAG recupera la validation accuracy ground-truth
    da NATS-Bench TSS e costruisce un DataFrame con proxy e GT.
    """
    api = load_nas201_api()

    rows = []
    for net, proxy_acc in zip(networks, accuracies):
        arch_str = networkdag_to_nb201_str(net)
        idx      = api.query_index_by_arch(arch_str)

        if idx < 0:
            gt_acc = None
        else:
            info   = api.get_more_info(idx, dataset, hp=hp, is_random=False)
            gt_acc = info.get('valid-accuracy', None)

        rows.append({
            'arch_str'  : arch_str,
            'Accuracy'  : proxy_acc * 100,   # supernet restituisce [0,1]
            'GT_Accuracy': gt_acc,
        })

    df = pd.DataFrame(rows)
    n_found = df['GT_Accuracy'].notna().sum()
    print(f"Architetture trovate nel benchmark: {n_found}/{len(networks)}")
    return df