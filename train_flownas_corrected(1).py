import argparse
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from models.flow import FlowNet
from models.nas201_models import VAE_dist, vae_accuracy_loss, vae_accuracy_loss_ws
from models.nas301_models import VAE_nas301, vae_accuracy_loss_nas301
from ws_universale.nb201 import nasbench201_strings_to_networkdags
from ws_universale.supernet import Supernet

from dataset_loader import (
    NASDatasetFactory,
    arch_to_tensor,
    load_nas201_api,
    load_nas301_performance_model,
    tensor_to_genotype,
)
from utils_functions.utils import (
    build_accuracy_pairs,
    decoded_x_to_nas201_arch,
    get_cifar10_loaders,
    query_nas201_accuracy,
    query_nas301_accuracy,
    random_nas201_arch,
    set_seed,
)


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().view(-1)
    return torch.as_tensor(x, dtype=torch.float32).view(-1)


def _dataset_to_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a dataset/subset containing pairs (x, y)."""
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=256, shuffle=False):
        xs.append(x.float().view(x.size(0), -1))
        ys.append(y.float().view(-1))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _sample_indices(length: int, n: int, seed: int) -> List[int]:
    if length <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[: min(n, length)].tolist()


def _decode_latents(model_VAE, z: torch.Tensor, input_dim: int) -> torch.Tensor:
    """Decode z and select the decoder tensor that represents the architecture."""
    decoded = model_VAE.decode(z)

    if isinstance(decoded, torch.Tensor):
        candidates = [decoded]
    elif isinstance(decoded, (tuple, list)):
        candidates = [item for item in decoded if isinstance(item, torch.Tensor)]
    else:
        raise TypeError(f"Unsupported decoder output: {type(decoded)!r}")

    for candidate in reversed(candidates):
        if candidate.size(0) == z.size(0) and candidate[0].numel() == input_dim:
            return candidate.view(candidate.size(0), -1)

    shapes = [tuple(item.shape) for item in candidates]
    raise RuntimeError(
        f"No decoder output has {input_dim} values per architecture. "
        f"Decoder tensor shapes: {shapes}"
    )


def _evaluate_nas201(
    architectures: Sequence[str],
    args,
    device: torch.device,
    api=None,
    supernet=None,
    train_image_loader=None,
    val_image_loader=None,
) -> torch.Tensor:
    if len(architectures) == 0:
        return torch.empty(0, dtype=torch.float32)

    if args.weight_sharing:
        network_dags = nasbench201_strings_to_networkdags(list(architectures))
        raw_accs = supernet.eval_subnets(
            networks=network_dags,
            train_loader=train_image_loader,
            eval_loader=val_image_loader,
            device=device,
            bn_batches=20,
            epochs=20,
            calibrate=True,
            M=4,
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
            use_label_smoothing=True,
            scheduler_factory=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=200, eta_min=1e-4
            ),
            optimizer_factory=lambda params: torch.optim.SGD(
                params,
                lr=0.025,
                momentum=0.9,
                weight_decay=3e-4,
                nesterov=True,
            ),
        )
        accs = [float(acc) for acc in raw_accs]
        # Accommodate either percentages or values already normalized to [0, 1].
        accs = [acc / 100.0 if acc > 1.5 else acc for acc in accs]
        return torch.tensor(accs, dtype=torch.float32)

    accs = []
    for arch_str in architectures:
        acc, _ = query_nas201_accuracy(
            api=api,
            arch_str=arch_str,
            dataset_name=args.dataset_name,
            hp=args.nas_hp,
            metric=args.nas_metric,
        )
        accs.append(0.0 if acc is None else float(acc) / 100.0)
    return torch.tensor(accs, dtype=torch.float32)


def _evaluate_nas301(genotypes, performance_model) -> torch.Tensor:
    accs = []
    for genotype in genotypes:
        acc, _ = query_nas301_accuracy(performance_model, genotype)
        accs.append(0.0 if acc is None else float(acc))
    return torch.tensor(accs, dtype=torch.float32)


def _select_next_population(
    current_x: torch.Tensor,
    current_archs: Sequence,
    current_accs: torch.Tensor,
    generated_x: torch.Tensor,
    generated_archs: Sequence,
    generated_accs: torch.Tensor,
    elite_fraction: float,
    population_size: int,
) -> Tuple[torch.Tensor, List, torch.Tensor, pd.DataFrame]:
    n_elite = min(
        len(current_archs),
        max(0, int(round(population_size * elite_fraction))),
    )
    n_generated = max(0, population_size - n_elite)

    current_order = torch.argsort(current_accs, descending=True)[:n_elite]
    generated_order = torch.argsort(generated_accs, descending=True)

    rows = []
    seen = set()

    for idx in current_order.tolist():
        key = str(current_archs[idx])
        if key not in seen:
            seen.add(key)
            rows.append((current_x[idx], current_archs[idx], current_accs[idx], "elite"))

    for idx in generated_order.tolist():
        if sum(row[3] == "flow" for row in rows) >= n_generated:
            break
        key = str(generated_archs[idx])
        if key not in seen:
            seen.add(key)
            rows.append(
                (generated_x[idx], generated_archs[idx], generated_accs[idx], "flow")
            )

    # If duplicates reduced the population, fill remaining slots with the best unused items.
    combined = []
    for source, xs, archs, accs in (
        ("flow", generated_x, generated_archs, generated_accs),
        ("elite", current_x, current_archs, current_accs),
    ):
        for idx in torch.argsort(accs, descending=True).tolist():
            combined.append((float(accs[idx]), source, idx, xs, archs, accs))

    for _, source, idx, xs, archs, accs in sorted(combined, reverse=True, key=lambda r: r[0]):
        if len(rows) >= population_size:
            break
        key = str(archs[idx])
        if key not in seen:
            seen.add(key)
            rows.append((xs[idx], archs[idx], accs[idx], source))

    if not rows:
        raise RuntimeError("The next population is empty.")

    next_x = torch.stack([row[0].float().view(-1) for row in rows])
    next_archs = [row[1] for row in rows]
    next_accs = torch.tensor([float(row[2]) for row in rows], dtype=torch.float32)
    next_df = pd.DataFrame(
        {
            "arch": next_archs,
            "acc": next_accs.tolist(),
            "source": [row[3] for row in rows],
        }
    ).sort_values("acc", ascending=False, ignore_index=True)

    return next_x, next_archs, next_accs, next_df


def run_training(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    benchmark_name = args.benchmark_name.upper()
    if args.weight_sharing and benchmark_name != "NAS201":
        raise ValueError("Weight sharing is currently supported only for NAS201.")

    api = None
    performance_model = None
    supernet = None
    train_image_loader = None
    val_image_loader = None

    if benchmark_name == "NAS201":
        api = getattr(args, "api", None) or load_nas201_api()
        model_VAE = VAE_dist(
            INPUT_DIM=80,
            LATENT_DIM=args.latent_dim,
            output_shape=(4, 4, 5),
        ).to(device)
        input_dim = 80

        if args.weight_sharing:
            supernet = Supernet().to(device)
            if args.dataset_name.lower() != "cifar10":
                raise ValueError("The current weight-sharing loaders support only CIFAR-10.")
            train_image_loader, val_image_loader = get_cifar10_loaders(
                batch_size=args.image_batch_size,
                num_workers=args.num_workers,
            )

    elif benchmark_name == "NAS301":
        performance_model = (
            args.performance_model
            if args.performance_model is not None
            else load_nas301_performance_model()
        )
        model_VAE = VAE_nas301(
            INPUT_DIM=504,
            LATENT_DIM=args.latent_dim,
            output_shape=(2, 7, 6, 6),
        ).to(device)
        input_dim = 504
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark_name}")

    flow = FlowNet(dim=args.latent_dim).to(device)

    # Dataset loading. In WS mode, random architectures are sufficient for VAE pretraining.
    if args.train_dataset is not None:
        if args.test_dataset is None:
            raise ValueError("test_dataset must be provided together with train_dataset.")
        train_dataset = args.train_dataset
        test_dataset = args.test_dataset
    elif benchmark_name == "NAS201":
        train_dataset, test_dataset, _, _ = NASDatasetFactory.create(
            benchmark_name="NAS201",
            api=api,
            dataset_name=args.dataset_name,
            metric=args.nas_metric,
            hp=args.nas_hp,
            flatten=True,
            normalize_y=True,
        )
    else:
        dataset, returned_model = NASDatasetFactory.create(
            benchmark_name="NAS301",
            performance_model=performance_model,
            n_samples=args.n_samples,
            flatten=True,
            normalize_y=True,
            seed=args.seed,
        )
        if returned_model is not None:
            performance_model = returned_model
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    print("\nPRETRAIN VAE")
    if args.weight_sharing:
        pretrain_archs = [random_nas201_arch() for _ in range(args.n_samples)]
        pretrain_x = torch.stack(
            [_as_tensor(arch_to_tensor(arch)) for arch in pretrain_archs]
        )
        pretrain_loader = DataLoader(
            TensorDataset(pretrain_x),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
        )
        pretrain_and_freeze_vae(
            model_VAE=model_VAE,
            pretrain_loader=pretrain_loader,
            loss_fn=vae_accuracy_loss_ws,
            vae_epochs=args.pretrain_vae_epochs,
            beta=args.beta,
            device=device,
            weight_sharing=True,
        )
    else:
        n_pretrain = max(1, int(len(train_dataset) * args.pretrain_fraction))
        pretrain_indices = _sample_indices(len(train_dataset), n_pretrain, args.seed)
        pretrain_loader = DataLoader(
            Subset(train_dataset, pretrain_indices),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
        )
        loss_fn = vae_accuracy_loss_nas301 if benchmark_name == "NAS301" else vae_accuracy_loss
        loss_kwargs = (
            {"pos_weight_value": args.pos_weight_value}
            if benchmark_name == "NAS301"
            else {}
        )
        pretrain_and_freeze_vae(
            model_VAE=model_VAE,
            pretrain_loader=pretrain_loader,
            loss_fn=loss_fn,
            vae_epochs=args.pretrain_vae_epochs,
            beta=args.beta,
            lambda_acc=args.lambda_acc,
            device=device,
            **loss_kwargs,
        )

    # Initial population.
    if args.weight_sharing:
        current_archs = [random_nas201_arch() for _ in range(args.N)]
        current_x = torch.stack(
            [_as_tensor(arch_to_tensor(arch)) for arch in current_archs]
        )
    else:
        all_x, _ = _dataset_to_tensors(train_dataset)
        initial_indices = _sample_indices(len(all_x), args.N, args.seed)
        current_x = all_x[initial_indices]
        if benchmark_name == "NAS201":
            current_archs = [decoded_x_to_nas201_arch(x) for x in current_x]
        else:
            current_archs = [tensor_to_genotype(x.flatten()) for x in current_x]

    history = {
        "epoch": [],
        "mean_acc": [],
        "std_acc": [],
        "min_acc": [],
        "max_acc": [],
        "population_size": [],
    }
    population_df = None

    for outer_epoch in range(args.outer_epochs):
        print(f"\nOUTER EPOCH {outer_epoch + 1}/{args.outer_epochs} {'=' * 10}")

        if benchmark_name == "NAS201":
            current_accs = _evaluate_nas201(
                current_archs,
                args,
                device,
                api=api,
                supernet=supernet,
                train_image_loader=train_image_loader,
                val_image_loader=val_image_loader,
            )
        else:
            current_accs = _evaluate_nas301(current_archs, performance_model)

        if current_accs.numel() == 0:
            print("No valid architecture in the current population.")
            break

        mean_acc = current_accs.mean().item()
        std_acc = current_accs.std(unbiased=False).item() if len(current_accs) > 1 else 0.0
        min_acc = current_accs.min().item()
        max_acc = current_accs.max().item()

        print(f"valid archs = {len(current_archs)}")
        print(f"mean acc    = {mean_acc:.4f}")
        print(f"std acc     = {std_acc:.4f}")
        print(f"min acc     = {min_acc:.4f}")
        print(f"max acc     = {max_acc:.4f}")

        history["epoch"].append(outer_epoch)
        history["mean_acc"].append(mean_acc)
        history["std_acc"].append(std_acc)
        history["min_acc"].append(min_acc)
        history["max_acc"].append(max_acc)
        history["population_size"].append(len(current_archs))

        population_loader = DataLoader(
            TensorDataset(current_x, current_accs),
            batch_size=args.batch_size,
            shuffle=True,
        )

        result = train_flow_and_generate(
            flow=flow,
            model_VAE=model_VAE,
            train_loader=population_loader,
            flow_epochs=args.flow_epochs,
            alpha=args.alpha,
            device=device,
            pair_k=args.pair_k,
            min_delta_acc=args.min_delta_acc,
            seed=args.seed + outer_epoch,
        )
        if result is None:
            print("Training stopped: no improving accuracy pair was found.")
            break

        z_new, _, _ = result
        with torch.no_grad():
            generated_x = _decode_latents(model_VAE, z_new, input_dim).cpu()

        if benchmark_name == "NAS201":
            generated_archs = [decoded_x_to_nas201_arch(x) for x in generated_x]
            generated_accs = _evaluate_nas201(
                generated_archs,
                args,
                device,
                api=api,
                supernet=supernet,
                train_image_loader=train_image_loader,
                val_image_loader=val_image_loader,
            )
        else:
            generated_archs = [tensor_to_genotype(x.flatten()) for x in generated_x]
            generated_accs = _evaluate_nas301(generated_archs, performance_model)

        current_x, current_archs, current_accs, population_df = _select_next_population(
            current_x=current_x,
            current_archs=current_archs,
            current_accs=current_accs,
            generated_x=generated_x,
            generated_archs=generated_archs,
            generated_accs=generated_accs,
            elite_fraction=args.elite_fraction,
            population_size=args.N,
        )

    if benchmark_name == "NAS301":
        return history, model_VAE, flow, test_dataset, performance_model, population_df
    return history, model_VAE, flow, test_dataset, api, population_df


def train_flow_and_generate(
    flow,
    model_VAE,
    train_loader,
    flow_epochs=100,
    alpha=0.5,
    device="cpu",
    pair_k=50,
    min_delta_acc=0.01,
    seed=42,
):
    """Encode the current population, train the flow and move latent points."""
    model_VAE.eval()
    flow = flow.to(device)

    z_all, y_all = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device).float()
            mu, _ = model_VAE.encode(x)
            z_all.append(mu.cpu())
            y_all.append(y.float().view(-1).cpu())

    z_all = torch.cat(z_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    pairs_x, pairs_target = build_accuracy_pairs(
        X=z_all,
        y=y_all,
        K=pair_k,
        min_delta_acc=min_delta_acc,
        seed=seed,
    )
    if len(pairs_x) == 0:
        return None

    pairs_loader = DataLoader(
        TensorDataset(pairs_x, pairs_target),
        batch_size=min(64, len(pairs_x)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.train()

    for _ in range(flow_epochs):
        for z_start, target_direction in pairs_loader:
            z_start = z_start.to(device)
            target_direction = target_direction.to(device)
            loss = F.mse_loss(flow(z_start), target_direction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    flow.eval()
    with torch.no_grad():
        z_start = z_all.to(device)
        z_new = z_start + alpha * flow(z_start)

    return z_new, z_all, y_all


def pretrain_and_freeze_vae(
    model_VAE,
    pretrain_loader,
    loss_fn,
    beta=0.0,
    lambda_acc=1.0,
    vae_epochs=300,
    device="cpu",
    early_stop=True,
    patience=20,
    min_delta=1e-5,
    loss_threshold=1e-4,
    lr=1e-3,
    weight_sharing=False,
    **loss_kwargs,
):
    model_VAE = model_VAE.to(device)
    optimizer = torch.optim.Adam(model_VAE.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(vae_epochs):
        model_VAE.train()
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "acc": 0.0}

        for batch in pretrain_loader:
            x = batch[0].to(device).float()
            y = None if len(batch) == 1 else batch[1].to(device).float().view(-1)

            outputs = model_VAE(x)
            if not isinstance(outputs, (tuple, list)):
                raise RuntimeError("The VAE forward method must return a tuple/list.")

            if weight_sharing:
                if len(outputs) < 4:
                    raise RuntimeError("WS VAE must return recon_logits, recon_probs, mu, logvar.")
                recon_logits, recon_probs, mu, logvar = outputs[:4]
                result = loss_fn(
                    recon_logits=recon_logits,
                    recon_probs=recon_probs,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    beta=beta,
                    **loss_kwargs,
                )
            else:
                if y is None:
                    raise RuntimeError("Supervised VAE pretraining requires accuracy targets.")
                if len(outputs) < 5:
                    raise RuntimeError(
                        "Supervised VAE must return recon_logits, recon_probs, mu, logvar, acc_pred."
                    )
                recon_logits, recon_probs, mu, logvar, acc_pred = outputs[:5]
                result = loss_fn(
                    recon_logits=recon_logits,
                    recon_probs=recon_probs,
                    x=x,
                    mu=mu,
                    logvar=logvar,
                    acc_pred=acc_pred,
                    true_acc=y,
                    beta=beta,
                    lambda_acc=lambda_acc,
                    **loss_kwargs,
                )

            if not isinstance(result, (tuple, list)) or len(result) < 3:
                raise RuntimeError("The VAE loss must return at least loss, recon_loss and kl.")

            loss, recon_loss, kl = result[:3]
            acc_loss = result[3] if len(result) > 3 else torch.zeros((), device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["recon"] += float(recon_loss.item())
            totals["kl"] += float(kl.item())
            totals["acc"] += float(acc_loss.item())

        batches = len(pretrain_loader)
        avg_loss = totals["loss"] / batches

        if epoch % 50 == 0:
            print(
                f"VAE pretrain epoch {epoch:03d} | "
                f"loss={avg_loss:.6f} | "
                f"recon={totals['recon'] / batches:.6f} | "
                f"kl={totals['kl'] / batches:.6f} | "
                f"acc_loss={totals['acc'] / batches:.6f}"
            )

        if early_stop:
            if avg_loss < loss_threshold:
                print(f"Early stopping at epoch {epoch}: loss={avg_loss:.6f}")
                break
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}: best_loss={best_loss:.6f}")
                break

    model_VAE.eval()
    for parameter in model_VAE.parameters():
        parameter.requires_grad = False

    print("VAE pretrained and frozen.")
    return model_VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Training FlowNAS")
    parser.add_argument("--outer_epochs", type=int, default=20)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=15000)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--pretrain_vae_epochs", type=int, default=300)
    parser.add_argument("--pretrain_fraction", type=float, default=1.0)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)
    parser.add_argument("--flow_epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pair_k", type=int, default=50)
    parser.add_argument("--min_delta_acc", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lambda_acc", type=float, default=5.0)
    parser.add_argument("--elite_fraction", type=float, default=0.1)
    parser.add_argument("--benchmark_name", type=str, default="NAS201")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--nas_hp", type=str, default="200")
    parser.add_argument("--nas_metric", type=str, default="test-accuracy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_dataset", default=None)
    parser.add_argument("--test_dataset", default=None)
    parser.add_argument("--performance_model", default=None)
    parser.add_argument("--api", default=None)
    parser.add_argument("--pos_weight_value", type=float, default=5.0)
    parser.add_argument("--weight_sharing", "--WS", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\nCONFIG")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    result = run_training(args)
    history, model_VAE, flow = result[:3]
