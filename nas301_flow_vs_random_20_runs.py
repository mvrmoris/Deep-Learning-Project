from __future__ import annotations
import copy
import gc
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset_loader import tensor_to_genotype
from utils import query_nas301_accuracy

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

def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """setting all seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def random_direction_same_norm(
    flow_direction: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    generate random direction with same norm as flow direction for fair comparison
    """
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
        ord=2,
        dim=1,
        keepdim=True,
    )
    random_norm = torch.linalg.vector_norm(
        random_direction,
        ord=2,
        dim=1,
        keepdim=True,
    )

    unit_random = random_direction / random_norm.clamp_min(eps)
    return unit_random * flow_norm


def _decode_latents_to_scores(
    model_VAE: torch.nn.Module,
    z: torch.Tensor,
    performance_model,
    device: torch.device,
) -> tuple[list, np.ndarray]:
    """
    Decodifica un batch di latenti NAS301 e restituisce genotipi e accuracy.

    Segue la stessa logica del progetto:
        decoded = model_VAE.decode(z)
        x_decoded = decoded[-1]
        genotype = tensor_to_genotype(x_decoded.view(-1))
        accuracy = query_nas301_accuracy(...)
    """
    model_VAE.eval()

    with torch.no_grad():
        decoded = model_VAE.decode(z.to(device).float())
        decoded_architectures = decoded[-1]

    decoded_architectures = decoded_architectures.detach().cpu()

    genotypes = []
    scores = []

    for x_decoded in decoded_architectures:
        genotype = tensor_to_genotype(x_decoded.view(-1))
        accuracy, _ = query_nas301_accuracy(
            performance_model=performance_model,
            arch=genotype,
        )
        genotypes.append(genotype)
        scores.append(np.nan if accuracy is None else float(accuracy))

    return genotypes, np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def compare_one_trained_model(
    model_VAE: torch.nn.Module,
    flow: torch.nn.Module,
    test_dataset,
    performance_model,
    alpha: float,
    batch_size: int,
    device: str | torch.device,
    run_index: int,
    seed: int,
    max_test_samples: Optional[int] = None,
) -> tuple[RunMetrics, pd.DataFrame]:
    """
    Confronta Flow e random sul test set per un singolo training.

    La baseline e' il decode di z_start, non l'accuracy originaria del dataset.
    Questo isola l'effetto dello spostamento latente dall'errore di ricostruzione
    del VAE.
    """
    device = torch.device(device)
    model_VAE = model_VAE.to(device).eval()
    flow = flow.to(device).eval()

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    rows: list[dict] = []
    processed = 0

    # Seed separato per il baseline random della run.
    random_generator = torch.Generator(device=device.type)
    random_generator.manual_seed(seed + 100_000)

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Il test_dataset deve restituire almeno (x, y)."
            )

        x, original_y = batch[0], batch[1]

        if max_test_samples is not None:
            remaining = max_test_samples - processed
            if remaining <= 0:
                break
            x = x[:remaining]
            original_y = original_y[:remaining]

        x = x.to(device).float()
        original_y = original_y.detach().cpu().float().view(-1).numpy()

        mu, _ = model_VAE.encode(x)
        z_start = mu

        flow_direction = flow(z_start)
        random_direction = random_direction_same_norm(
            flow_direction=flow_direction,
            generator=random_generator,
        )

        z_flow = z_start + float(alpha) * flow_direction
        z_random = z_start + float(alpha) * random_direction

        start_genotypes, start_scores = _decode_latents_to_scores(
            model_VAE=model_VAE,
            z=z_start,
            performance_model=performance_model,
            device=device,
        )
        flow_genotypes, flow_scores = _decode_latents_to_scores(
            model_VAE=model_VAE,
            z=z_flow,
            performance_model=performance_model,
            device=device,
        )
        random_genotypes, random_scores = _decode_latents_to_scores(
            model_VAE=model_VAE,
            z=z_random,
            performance_model=performance_model,
            device=device,
        )

        flow_norms = torch.linalg.vector_norm(
            flow_direction, dim=1
        ).detach().cpu().numpy()
        random_norms = torch.linalg.vector_norm(
            random_direction, dim=1
        ).detach().cpu().numpy()

        for i in range(len(start_scores)):
            start_acc = start_scores[i]
            flow_acc = flow_scores[i]
            random_acc = random_scores[i]

            flow_improvement = flow_acc - start_acc
            random_improvement = random_acc - start_acc
            flow_advantage = flow_acc - random_acc

            rows.append(
                {
                    "run": run_index,
                    "seed": seed,
                    "sample_index": processed + i,
                    "dataset_accuracy": float(original_y[i]),
                    "start_accuracy": start_acc,
                    "flow_accuracy": flow_acc,
                    "random_accuracy": random_acc,
                    "flow_improvement": flow_improvement,
                    "random_improvement": random_improvement,
                    "flow_advantage": flow_advantage,
                    "flow_success": bool(flow_improvement > 0),
                    "random_success": bool(random_improvement > 0),
                    "flow_better": bool(flow_advantage > 0),
                    "tie": bool(np.isclose(flow_advantage, 0.0)),
                    "flow_direction_norm": float(flow_norms[i]),
                    "random_direction_norm": float(random_norms[i]),
                    "start_genotype": str(start_genotypes[i]),
                    "flow_genotype": str(flow_genotypes[i]),
                    "random_genotype": str(random_genotypes[i]),
                }
            )

        processed += len(start_scores)

    samples_df = pd.DataFrame(rows)

    if samples_df.empty:
        raise RuntimeError("Nessun campione test e' stato valutato.")

    valid_mask = np.isfinite(
        samples_df[
            ["start_accuracy", "flow_accuracy", "random_accuracy"]
        ].to_numpy()
    ).all(axis=1)
    samples_df = samples_df.loc[valid_mask].reset_index(drop=True)

    if samples_df.empty:
        raise RuntimeError("Tutte le valutazioni NAS301 sono risultate non valide.")

    metrics = RunMetrics(
        run=run_index,
        seed=seed,
        n_samples=len(samples_df),
        mean_start_accuracy=float(samples_df["start_accuracy"].mean()),
        mean_flow_accuracy=float(samples_df["flow_accuracy"].mean()),
        mean_random_accuracy=float(samples_df["random_accuracy"].mean()),
        mean_flow_improvement=float(samples_df["flow_improvement"].mean()),
        mean_random_improvement=float(samples_df["random_improvement"].mean()),
        mean_flow_advantage=float(samples_df["flow_advantage"].mean()),
        flow_success_rate=float(samples_df["flow_success"].mean()),
        random_success_rate=float(samples_df["random_success"].mean()),
        flow_better_rate=float(samples_df["flow_better"].mean()),
        tie_rate=float(samples_df["tie"].mean()),
        mean_flow_direction_norm=float(samples_df["flow_direction_norm"].mean()),
        mean_random_direction_norm=float(samples_df["random_direction_norm"].mean()),
    )

    return metrics, samples_df


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Media, deviazione standard e standard error sulle run."""
    metric_columns = [
        "mean_start_accuracy",
        "mean_flow_accuracy",
        "mean_random_accuracy",
        "mean_flow_improvement",
        "mean_random_improvement",
        "mean_flow_advantage",
        "flow_success_rate",
        "random_success_rate",
        "flow_better_rate",
        "tie_rate",
        "mean_flow_direction_norm",
        "mean_random_direction_norm",
    ]

    summary_rows = []
    n_runs = len(results_df)

    for column in metric_columns:
        values = results_df[column].astype(float)
        std = float(values.std(ddof=1)) if n_runs > 1 else 0.0
        summary_rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std_across_runs": std,
                "sem_across_runs": std / np.sqrt(n_runs) if n_runs > 0 else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(summary_rows)


def run_flow_vs_random_experiment(
    base_args,
    run_training_fn: Callable,
    n_runs: int = 20,
    max_test_samples: Optional[int] = None,
    initial_seed: int = 42,
    output_dir: str = "results_flow_vs_random",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    train and comparison between different trainings
    """

    os.makedirs(output_dir, exist_ok=True)

    all_run_metrics: list[dict] = []
    all_sample_frames: list[pd.DataFrame] = []

    for run_zero_based in range(n_runs):
        run_index = run_zero_based + 1
        seed = initial_seed + run_zero_based

        print("\n" + "=" * 80)
        print(f"RUN {run_index}/{n_runs} | seed={seed}")
        print("=" * 80)

        set_all_seeds(seed)

        run_args = copy.copy(base_args)
        run_args.seed = seed

        (
            history,
            model_VAE,
            flow,
            test_dataset,
            performance_model,
        ) = run_training_fn(run_args)

        metrics, run_samples_df = compare_one_trained_model(
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

        metric_dict = metrics.__dict__
        all_run_metrics.append(metric_dict)
        all_sample_frames.append(run_samples_df)

        print(f"Campioni validi:             {metrics.n_samples}")
        print(f"Baseline decoded media:      {metrics.mean_start_accuracy:.6f}")
        print(f"Miglioramento Flow:          {metrics.mean_flow_improvement:+.6f}")
        print(f"Miglioramento Random:        {metrics.mean_random_improvement:+.6f}")
        print(f"Vantaggio Flow - Random:     {metrics.mean_flow_advantage:+.6f}")
        print(f"Flow migliora baseline:      {100 * metrics.flow_success_rate:.2f}%")
        print(f"Random migliora baseline:    {100 * metrics.random_success_rate:.2f}%")
        print(f"Flow migliore del Random:    {100 * metrics.flow_better_rate:.2f}%")
        print(
            "Norma media Flow/Random:    "
            f"{metrics.mean_flow_direction_norm:.6f} / "
            f"{metrics.mean_random_direction_norm:.6f}"
        )

        # Salvataggio incrementale: non perdi le run gia' completate.
        pd.DataFrame(all_run_metrics).to_csv(
            os.path.join(output_dir, "run_results_partial.csv"),
            index=False,
        )
        pd.concat(all_sample_frames, ignore_index=True).to_csv(
            os.path.join(output_dir, "sample_results_partial.csv"),
            index=False,
        )

        del history, model_VAE, flow
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_run_metrics)
    samples_df = pd.concat(all_sample_frames, ignore_index=True)
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

    print("\n" + "=" * 80)
    print(f"RISULTATI AGGREGATI SU {n_runs} TRAINING")
    print("=" * 80)

    def report(column: str, label: str, percentage: bool = False) -> None:
        mean = results_df[column].mean()
        std = results_df[column].std(ddof=1) if len(results_df) > 1 else 0.0
        factor = 100.0 if percentage else 1.0
        suffix = "%" if percentage else ""
        print(f"{label:<31} {factor * mean:+.6f} ± {factor * std:.6f}{suffix}")

    report("mean_start_accuracy", "Baseline decoded")
    report("mean_flow_improvement", "Miglioramento Flow")
    report("mean_random_improvement", "Miglioramento Random")
    report("mean_flow_advantage", "Vantaggio Flow - Random")
    report("flow_success_rate", "Success rate Flow", percentage=True)
    report("random_success_rate", "Success rate Random", percentage=True)
    report("flow_better_rate", "Flow migliore del Random", percentage=True)

    print(f"\nCSV salvati in: {os.path.abspath(output_dir)}")

    return results_df, samples_df, summary_df
