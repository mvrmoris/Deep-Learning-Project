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

from utils_functions.utils import decode_population_nas301, set_seed


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
def compare_one_trained_model(
    model_VAE,
    flow,
    test_dataset,
    performance_model,
    alpha,
    batch_size,
    device,
    run_index,
    seed,
    max_test_samples=None,
):
    device = torch.device(device)
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

    rows = []
    processed = 0

    for x, dataset_accuracy in loader:
        if max_test_samples is not None:
            remaining = max_test_samples - processed
            if remaining <= 0:
                break

            x = x[:remaining]
            dataset_accuracy = dataset_accuracy[:remaining]

        x = x.to(device).float()
        dataset_accuracy = dataset_accuracy.cpu().view(-1).numpy()

        z_start, _ = model_VAE.encode(x)
        flow_direction = flow(z_start)
        random_direction = random_direction_same_norm(
            flow_direction,
            generator,
        )

        latent_batches = {
            "start": z_start,
            "flow": z_start + alpha * flow_direction,
            "random": z_start + alpha * random_direction,
        }

        decoded = {
            name: decode_population_nas301(
                model_VAE,
                z,
                performance_model,
                device,
            )
            for name, z in latent_batches.items()
        }

        start_genotypes, start_scores, _ = decoded["start"]
        flow_genotypes, flow_scores, _ = decoded["flow"]
        random_genotypes, random_scores, _ = decoded["random"]

        start_scores = np.asarray(start_scores, dtype=float)
        flow_scores = np.asarray(flow_scores, dtype=float)
        random_scores = np.asarray(random_scores, dtype=float)

        flow_norms = torch.linalg.vector_norm(
            flow_direction,
            dim=1,
        ).cpu().numpy()

        random_norms = torch.linalg.vector_norm(
            random_direction,
            dim=1,
        ).cpu().numpy()

        n = min(
            len(start_scores),
            len(flow_scores),
            len(random_scores),
            len(dataset_accuracy),
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
                "start_genotype": str(start_genotypes[i]),
                "flow_genotype": str(flow_genotypes[i]),
                "random_genotype": str(random_genotypes[i]),
            })

        processed += n

    samples_df = pd.DataFrame(rows)

    if samples_df.empty:
        raise RuntimeError("Nessun campione valutato.")

    valid = np.isfinite(
        samples_df[
            ["start_accuracy", "flow_accuracy", "random_accuracy"]
        ].to_numpy()
    ).all(axis=1)

    samples_df = samples_df.loc[valid].reset_index(drop=True)

    if samples_df.empty:
        raise RuntimeError("Nessuna valutazione NAS301 valida.")

    metrics = RunMetrics(
        run=run_index,
        seed=seed,
        n_samples=len(samples_df),
        mean_start_accuracy=float(samples_df.start_accuracy.mean()),
        mean_flow_accuracy=float(samples_df.flow_accuracy.mean()),
        mean_random_accuracy=float(samples_df.random_accuracy.mean()),
        mean_flow_improvement=float(samples_df.flow_improvement.mean()),
        mean_random_improvement=float(samples_df.random_improvement.mean()),
        mean_flow_advantage=float(samples_df.flow_advantage.mean()),
        flow_success_rate=float(samples_df.flow_success.mean()),
        random_success_rate=float(samples_df.random_success.mean()),
        flow_better_rate=float(samples_df.flow_better.mean()),
        tie_rate=float(samples_df.tie.mean()),
        mean_flow_direction_norm=float(samples_df.flow_direction_norm.mean()),
        mean_random_direction_norm=float(samples_df.random_direction_norm.mean()),
    )

    return metrics, samples_df


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"run", "seed", "n_samples"}
    n_runs = len(results_df)
    rows = []

    for column in results_df.columns:
        if column in excluded:
            continue

        values = results_df[column].astype(float)
        std = float(values.std(ddof=1)) if n_runs > 1 else 0.0

        rows.append({
            "metric": column,
            "mean": float(values.mean()),
            "std_across_runs": std,
            "sem_across_runs": std / np.sqrt(n_runs),
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
    os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    sample_frames = []

    for run_index in range(1, n_runs + 1):
        seed = initial_seed + run_index - 1

        print(f"\n{'=' * 80}")
        print(f"RUN {run_index}/{n_runs} | seed={seed}")
        print("=" * 80)

        set_seed(seed)

        run_args = copy.copy(base_args)
        run_args.seed = seed

        (
            history,
            model_VAE,
            flow,
            test_dataset,
            performance_model,
        ) = run_training_fn(run_args)

        metrics, samples_df = compare_one_trained_model(
            model_VAE=model_VAE,
            flow=flow,
            test_dataset=test_dataset,
            performance_model=performance_model,
            alpha=run_args.alpha,
            batch_size=run_args.batch_size,
            device=run_args.device,
            run_index=run_index,
            seed=seed,
            max_test_samples=max_test_samples,
        )

        metric_rows.append(asdict(metrics))
        sample_frames.append(samples_df)

        print(f"Campioni validi:          {metrics.n_samples}")
        print(f"Miglioramento Flow:       {metrics.mean_flow_improvement:+.6f}")
        print(f"Miglioramento Random:     {metrics.mean_random_improvement:+.6f}")
        print(f"Vantaggio Flow-Random:    {metrics.mean_flow_advantage:+.6f}")
        print(f"Flow migliore del Random: {100 * metrics.flow_better_rate:.2f}%")

        pd.DataFrame(metric_rows).to_csv(
            os.path.join(output_dir, "run_results_partial.csv"),
            index=False,
        )
        pd.concat(sample_frames, ignore_index=True).to_csv(
            os.path.join(output_dir, "sample_results_partial.csv"),
            index=False,
        )

        del history, model_VAE, flow
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(metric_rows)
    samples_df = pd.concat(sample_frames, ignore_index=True)
    summary_df = build_summary(results_df)

    results_df.to_csv(
        os.path.join(output_dir, "run_results.csv"),
        index=False,
    )
    samples_df.to_csv(
        os.path.join(output_dir, "sample_results.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(output_dir, "summary.csv"),
        index=False,
    )

    return results_df, samples_df, summary_df
