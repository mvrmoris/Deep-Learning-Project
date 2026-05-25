import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


OPS = {
    "nor_conv_3x3": 0,
    "nor_conv_1x1": 1,
    "skip_connect": 2,
    "avg_pool_3x3": 3,
    "none": 4,
    "zeroize": 5,
}


DATASET_NAME_MAP = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet16-120": "ImageNet16-120",
}



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
        flatten=True,
        normalize_y=True,
    ):
        self.api = api
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        self.metric = metric
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
            is_random=False
        )

        acc = info.get(self.metric)

        if acc is None:
            raise ValueError(
                f"Metrica '{self.metric}' non trovata per idx={idx}, "
                f"dataset={self.dataset_name}. "
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
    
class NAS301TorchDatasetBuilder():
    #TO DO 
    pass

class NASDatasetFactory:

    @staticmethod
    def create(
        benchmark_name,
        *,
        api=None,
        performance_model=None,
        dataset_name="cifar10",
        metric="test-accuracy",
        n_samples=5000,
        flatten=True,
        normalize_y=True,
        seed=42,
    ):
        benchmark_name = benchmark_name.upper()

        if benchmark_name == "NAS201":
            if api is None:
                raise ValueError("Per NAS201 devi passare api.")

            builder = NAS201TorchDatasetBuilder(
                api=api,
                dataset_name=dataset_name,
                metric=metric,
                flatten=flatten,
                normalize_y=normalize_y,
            )

            return builder.build_dataset()

        elif benchmark_name == "NAS301":
            if performance_model is None:
                raise ValueError("Per NAS301 devi passare performance_model.")

            builder = NAS301TorchDatasetBuilder(
                performance_model=performance_model,
                n_samples=n_samples,
                flatten=flatten,
                normalize_y=normalize_y,
                seed=seed,
            )

            return builder.build_dataset()

        else:
            raise ValueError(
                f"Benchmark non supportato: {benchmark_name}. "
                "Scegli tra NAS201 e NAS301."
            )