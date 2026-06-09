import pandas as pd
from ws_universale.nb201 import networkdag_to_nb201_str
import numpy as np 
import re
import os
import tarfile
import torch
from nats_bench import create
from torch.utils.data import TensorDataset, DataLoader, random_split

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
#NAS201 dataset utils
def arch_to_tensor(arch_str):
    """Convert NAS201 string into tensor"""
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
import os
import tarfile
import shutil
import gdown

from nats_bench import create


def load_nas201_api(
    datasets_dir=None,
    tar_name="NATS-tss-v1_0-3ffb9-simple.tar",
    extracted_name="NATS-tss-v1_0-3ffb9-simple",
    fast_mode=True,
    verbose=False,
):
    """Download, extract and load the NAS-Bench-201 API."""

    datasets_dir = datasets_dir or os.path.dirname(
        os.path.abspath(__file__)
    )
    os.makedirs(datasets_dir, exist_ok=True)

    tar_path = os.path.join(datasets_dir, tar_name)
    dataset_path = os.path.join(datasets_dir, extracted_name)

    # Official NATS-Bench Google Drive file ID
    file_id = "17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"

    if not os.path.exists(dataset_path):

        if not os.path.isfile(tar_path):
            print("NAS-Bench-201 dataset not found. Downloading...")

            gdown.download(
                id=file_id,
                output=tar_path,
                quiet=False,
            )

            if not os.path.isfile(tar_path):
                raise RuntimeError(
                    "NAS-Bench-201 download failed."
                )

        print("Extracting NAS-Bench-201 dataset...")

        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(datasets_dir)

        if not os.path.exists(dataset_path):
            raise RuntimeError(
                f"Extraction completed, but the expected directory "
                f"was not found: {dataset_path}"
            )

        print(f"Dataset available at: {dataset_path}")

    return create(
        dataset_path,
        "tss",
        fast_mode=fast_mode,
        verbose=verbose,
    )

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

            return builder.build_loaders()

        elif benchmark_name == "NAS301":
            # Import eseguito solo quando serve NAS301
            from datasets.dataset_loader_nas301 import (
                NAS301TorchDatasetBuilder,
                load_nas301_performance_model,
            )

            if performance_model is None:
                performance_model = load_nas301_performance_model()

            builder = NAS301TorchDatasetBuilder(
                performance_model=performance_model,
                n_samples=n_samples,
                flatten=flatten,
                normalize_y=normalize_y,
                seed=seed,
            )

            (
                train_dataset,
                test_dataset,
                train_loader,
                test_loader,
            ) = builder.build_loaders()

            return (
                train_dataset,
                test_dataset,
                train_loader,
                test_loader,
                performance_model,
            )

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