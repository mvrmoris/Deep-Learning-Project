from __future__ import annotations

import copy
import gc
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils_functions.utils import (
    decode_population_nas301,
    decoded_x_to_nas201_arch,
    query_nas201_accuracy,
    set_seed,
)


@dataclass
class RunMetrics:
    run: int
    seed: int
    n_samples: int
    mean_start_accuracy: float
    mean_flow_accuracy: float
    mean_random_accuracy: float
    mean_flow_improvement: float
    mean_random_improvement: float
    mean_flow_advantage: float
    flow_success_rate: float
    random_success_rate: float
    flow_better_rate: float
    tie_rate: float
    mean_flow_direction_norm: float
    mean_random_direction_norm: float


def random_direction_same_norm(
    flow_direction: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Genera una direzione casuale con la stessa norma della direzione Flow."""

    if flow_direction.ndim != 2:
        raise ValueError(
            "flow_direction deve avere shape [batch_size, latent_dim], "
            f"ricevuta {tuple(flow_direction.shape)}"
        )

    random_direction = torch.randn(
        flow_direction.shape,
        dtype=flow_direction.dtype,
        device=flow_direction.device,
        generator=generator,
    )

    flow_norm = torch.linalg.vector_norm(
        flow_direction,
        dim=1,
        keepdim=True,
    )

    random_norm = torch.linalg.vector_norm(
        random_direction,
        dim=1,
        keepdim=True,
    )

    return random_direction / random_norm.clamp_min(eps) * flow_norm


@torch.no_grad()
def decode_population_nas201(
    model_VAE,
    z,
    api,
    device,
    dataset_name,
    hp="200",
    metric="test-accuracy",
):
    """
    Decodifica i latenti NAS201 in stringhe architetturali
    e interroga la relativa accuracy tramite API.
    """

    model_VAE.eval()

    decoded = model_VAE.decode(
        z.to(device).float()
    )

    decoded_architectures = decoded[-1].detach().cpu()

    architectures = []
    accuracies = []
    infos = []

    for x_decoded in decoded_architectures:
        # Il decoder NAS201 produce tipicamente shape (4, 4, 5),
        # mentre decoded_x_to_nas201_arch usa la disposizione (5, 4, 4).
        if x_decoded.ndim == 3 and tuple(x_decoded.shape) == (4, 4, 5):
            x_for_converter = x_decoded.permute(2, 0, 1).flatten()
        else:
            x_for_converter = x_decoded.flatten()

        arch_str = decoded_x_to_nas201_arch(
            x_for_converter
        )

        accuracy, info = query_nas201_accuracy(
            api=api,
            arch_str=arch_str,
            dataset_name=dataset_name,
            hp=hp,
            metric=metric,
        )

        architectures.append(arch_str)
        accuracies.append(
            np.nan
            if accuracy is None
            else float(accuracy) / 100.0
        )
        infos.append(info)

    return architectures, accuracies, infos


@torch.no_grad()
def compare_one_trained_model(
    model_VAE,
    flow,
    test_dataset,
    evaluator,
    alpha,
    batch_size,
    device,
    run_index,
    seed,
    benchmark_name,
    dataset_name=None,
    nas_hp="200",
    nas_metric="test-accuracy",
    max_test_samples=None,
):
    """
    Confronta Flow e una direzione casuale di uguale norma.

    evaluator:
        - performance_model per NAS301
        - api per NAS201
    """

    device = torch.device(device)
    benchmark_name = benchmark_name.upper()

    model_VAE = model_VAE.to(device).eval()
    flow = flow.to(device).eval()

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    generator = torch.Generator(
        device=device.type
    ).manual_seed(seed + 100_000)

    if benchmark_name == "NAS301":

        def decode_scores(z):
            return decode_population_nas301(
                model_VAE,
                z,
                evaluator,
                device,
            )

    elif benchmark_name == "NAS201":

        def decode_scores(z):
            return decode_population_nas201(
                model_VAE=model_VAE,
                z=z,
                api=evaluator,
                device=device,
                dataset_name=dataset_name,
                hp=nas_hp,
                metric=nas_metric,
            )

    else:
        raise ValueError(
            f"Benchmark non supportato: {benchmark_name}"
        )

    rows = []
    processed = 0

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Il test_dataset deve restituire almeno (x, y)."
            )

        x, dataset_accuracy = batch[0], batch[1]

        if max_test_samples is not None:
            remaining = max_test_samples - processed

            if remaining <= 0:
                break

            x = x[:remaining]
            dataset_accuracy = dataset_accuracy[:remaining]

        x = x.to(device).float()

        dataset_accuracy = (
            dataset_accuracy
            .detach()
            .cpu()
            .float()
            .view(-1)
            .numpy()
        )

        z_start, _ = model_VAE.encode(x)

        flow_direction = flow(z_start)

        random_direction = random_direction_same_norm(
            flow_direction,
            generator,
        )

        z_flow = z_start + float(alpha) * flow_direction
        z_random = z_start + float(alpha) * random_direction

        start_archs, start_scores, _ = decode_scores(z_start)
        flow_archs, flow_scores, _ = decode_scores(z_flow)
        random_archs, random_scores, _ = decode_scores(z_random)

        start_scores = np.asarray(start_scores, dtype=np.float64)
        flow_scores = np.asarray(flow_scores, dtype=np.float64)
        random_scores = np.asarray(random_scores, dtype=np.float64)

        flow_norms = (
            torch.linalg.vector_norm(
                flow_direction,
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        random_norms = (
            torch.linalg.vector_norm(
                random_direction,
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        n = min(
            len(start_scores),
            len(flow_scores),
            len(random_scores),
            len(dataset_accuracy),
            len(start_archs),
            len(flow_archs),
            len(random_archs),
        )

        for i in range(n):
            start_acc = float(start_scores[i])
            flow_acc = float(flow_scores[i])
            random_acc = float(random_scores[i])

            flow_improvement = flow_acc - start_acc
            random_improvement = random_acc - start_acc
            flow_advantage = flow_acc - random_acc

            rows.append({
                "run": run_index,
                "seed": seed,
                "sample_index": processed + i,
                "dataset_accuracy": float(dataset_accuracy[i]),
                "start_accuracy": start_acc,
                "flow_accuracy": flow_acc,
                "random_accuracy": random_acc,
                "flow_improvement": flow_improvement,
                "random_improvement": random_improvement,
                "flow_advantage": flow_advantage,
                "flow_success": flow_improvement > 0,
                "random_success": random_improvement > 0,
                "flow_better": flow_advantage > 0,
                "tie": np.isclose(flow_advantage, 0.0),
                "flow_direction_norm": float(flow_norms[i]),
                "random_direction_norm": float(random_norms[i]),
                "start_arch": str(start_archs[i]),
                "flow_arch": str(flow_archs[i]),
                "random_arch": str(random_archs[i]),
            })

        processed += n

    samples_df = pd.DataFrame(rows)

    if samples_df.empty:
        raise RuntimeError("Nessun campione valutato.")

    valid = np.isfinite(
        samples_df[
            [
                "start_accuracy",
                "flow_accuracy",
                "random_accuracy",
            ]
        ].to_numpy()
    ).all(axis=1)

    samples_df = (
        samples_df.loc[valid]
        .reset_index(drop=True)
    )

    if samples_df.empty:
        raise RuntimeError(
            f"Nessuna valutazione valida per {benchmark_name}."
        )

    metrics = RunMetrics(
        run=run_index,
        seed=seed,
        n_samples=len(samples_df),
        mean_start_accuracy=float(
            samples_df["start_accuracy"].mean()
        ),
        mean_flow_accuracy=float(
            samples_df["flow_accuracy"].mean()
        ),
        mean_random_accuracy=float(
            samples_df["random_accuracy"].mean()
        ),
        mean_flow_improvement=float(
            samples_df["flow_improvement"].mean()
        ),
        mean_random_improvement=float(
            samples_df["random_improvement"].mean()
        ),
        mean_flow_advantage=float(
            samples_df["flow_advantage"].mean()
        ),
        flow_success_rate=float(
            samples_df["flow_success"].mean()
        ),
        random_success_rate=float(
            samples_df["random_success"].mean()
        ),
        flow_better_rate=float(
            samples_df["flow_better"].mean()
        ),
        tie_rate=float(
            samples_df["tie"].mean()
        ),
        mean_flow_direction_norm=float(
            samples_df["flow_direction_norm"].mean()
        ),
        mean_random_direction_norm=float(
            samples_df["random_direction_norm"].mean()
        ),
    )

    return metrics, samples_df


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calcola media, deviazione standard, SEM, minimo e massimo."""

    excluded = {"run", "seed", "n_samples"}
    n_runs = len(results_df)
    rows = []

    for column in results_df.columns:
        if column in excluded:
            continue

        values = results_df[column].astype(float)

        std = (
            float(values.std(ddof=1))
            if n_runs > 1
            else 0.0
        )

        rows.append({
            "metric": column,
            "mean": float(values.mean()),
            "std_across_runs": std,
            "sem_across_runs": (
                std / np.sqrt(n_runs)
                if n_runs > 0
                else np.nan
            ),
            "min": float(values.min()),
            "max": float(values.max()),
        })

    return pd.DataFrame(rows)


def run_flow_vs_random_experiment(
    base_args,
    run_training_fn: Callable,
    n_runs: int = 20,
    max_test_samples: Optional[int] = None,
    initial_seed: int = 42,
    output_dir: str = "results_flow_vs_random",
):
    """
    Esegue più training e confronta Flow con una baseline casuale.

    Compatibile con:
        - NAS201
        - NAS301
    """

    os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    sample_frames = []

    for run_index in range(1, n_runs + 1):
        seed = initial_seed + run_index - 1

        print("\n" + "=" * 80)
        print(
            f"RUN {run_index}/{n_runs} | "
            f"{base_args.benchmark_name.upper()} | "
            f"{getattr(base_args, 'dataset_name', '')} | "
            f"seed={seed}"
        )
        print("=" * 80)

        set_seed(seed)

        run_args = copy.copy(base_args)
        run_args.seed = seed

        (
            history,
            model_VAE,
            flow,
            test_dataset,
            evaluator,
        ) = run_training_fn(run_args)

        metrics, samples_df = compare_one_trained_model(
            model_VAE=model_VAE,
            flow=flow,
            test_dataset=test_dataset,
            evaluator=evaluator,
            alpha=run_args.alpha,
            batch_size=run_args.batch_size,
            device=run_args.device,
            run_index=run_index,
            seed=seed,
            benchmark_name=run_args.benchmark_name,
            dataset_name=getattr(
                run_args,
                "dataset_name",
                None,
            ),
            nas_hp=getattr(
                run_args,
                "nas_hp",
                "200",
            ),
            nas_metric=getattr(
                run_args,
                "nas_metric",
                "test-accuracy",
            ),
            max_test_samples=max_test_samples,
        )

        metric_rows.append(asdict(metrics))
        sample_frames.append(samples_df)

        print(f"Campioni validi:          {metrics.n_samples}")
        print(f"Baseline media:           {metrics.mean_start_accuracy:.6f}")
        print(f"Miglioramento Flow:       {metrics.mean_flow_improvement:+.6f}")
        print(f"Miglioramento Random:     {metrics.mean_random_improvement:+.6f}")
        print(f"Vantaggio Flow-Random:    {metrics.mean_flow_advantage:+.6f}")
        print(f"Flow migliore del Random: {100 * metrics.flow_better_rate:.2f}%")

        pd.DataFrame(metric_rows).to_csv(
            os.path.join(
                output_dir,
                "run_results_partial.csv",
            ),
            index=False,
        )

        pd.concat(
            sample_frames,
            ignore_index=True,
        ).to_csv(
            os.path.join(
                output_dir,
                "sample_results_partial.csv",
            ),
            index=False,
        )

        del history, model_VAE, flow
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(metric_rows)

    samples_df = pd.concat(
        sample_frames,
        ignore_index=True,
    )

    summary_df = build_summary(results_df)

    results_df.to_csv(
        os.path.join(
            output_dir,
            "run_results.csv",
        ),
        index=False,
    )

    samples_df.to_csv(
        os.path.join(
            output_dir,
            "sample_results.csv",
        ),
        index=False,
    )

    summary_df.to_csv(
        os.path.join(
            output_dir,
            "summary.csv",
        ),
        index=False,
    )

    return results_df, samples_df, summary_df
