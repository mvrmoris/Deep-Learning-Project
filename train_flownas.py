
import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset
from latent_space.model import VAE_dist, FlowNet
from latent_space.dataset_loader import NASDatasetFactory,load_nas201_api
import os
import tarfile
from nats_bench import create
from latent_space.utils import (
    set_seed,
    pretrain_and_freeze_vae,
    generate_archs,
    train_one_epoch,
    decoded_x_to_nas201_arch,
    query_nas201_accuracy,
    arch_to_tensor
)


def run_training(args):

    DEVICE = torch.device(args.device)
    set_seed(args.seed)
    api = load_nas201_api()

    #models
    model_VAE = VAE_dist(LATENT_DIM=args.latent_dim).to(DEVICE)
    flow = FlowNet(dim=args.latent_dim).to(DEVICE)
    #Dataset
    dataset = NASDatasetFactory.create(
        api = api,
        benchmark_name="NAS201",
        dataset_name=args.dataset_name,
        metric=args.nas_metric,
        hp=args.nas_hp,
        flatten=True,
        normalize_y=True,
        )

    print("Numero architetture:", len(dataset))
    #dataset and dataloader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    # training
    #1. VAE pretrain 

    print("\n PRETRAIN VAE")
    n_pretrain = int(train_size * args.pretrain_fraction)

    generator = torch.Generator().manual_seed(args.seed)

    indices = torch.randperm(
        train_size,
        generator=generator
    )[:n_pretrain].tolist()

    pretrain_subset = Subset(
        train_dataset,
        indices
    )
    pretrain_loader = DataLoader(
        pretrain_subset,
        batch_size=args.pretrain_batch_size,
        shuffle=True
    )

    pretrain_and_freeze_vae(
        model_VAE=model_VAE,
        pretrain_loader=pretrain_loader,
        vae_epochs=args.pretrain_vae_epochs,
        beta=args.beta,
        lambda_acc=args.lambda_acc,
        DEVICE=DEVICE
    )

    #initial pool of architectures
    train_loader = generate_archs(
        dataset=train_dataset,
        N=args.N
    )

    history = {
        "epoch": [],
        "mean_acc": [],
        "std_acc": [],
        "min_acc": [],
        "max_acc": [],
        "population_size": []
    }
    #training loop
    for outer_epoch in range(args.outer_epochs):

        print(
            f"\n OUTER EPOCH "
            f"{outer_epoch + 1}/{args.outer_epochs} =========="
        )
        #1. training Flow
        result = train_one_epoch(
            flow=flow,
            model_VAE=model_VAE,
            train_loader=train_loader,
            beta=args.beta,
            lambda_acc=args.lambda_acc,
            vae_epochs=args.vae_epochs,
            flow_epochs=args.flow_epochs,
            alpha=args.alpha,
            DEVICE=DEVICE,
            train_vae=False,
        )

        if result is None:
            print("Training interrotto: nessuna coppia valida trovata.")
            break

        z_new, z_all, y_all, pairs_info = result

        #2. decoding new architectures
        model_VAE.eval()

        with torch.no_grad():
            recon_logits_new, recon_probs_new = model_VAE.decode(
                z_new.to(DEVICE).float()
            )

        recon_probs_new = recon_probs_new.detach().cpu()

        #into NAS201 strings to query accuracy
        new_archs = []
        new_accs = []
        new_infos = []

        for i in range(recon_probs_new.shape[0]):

            x_decoded = recon_probs_new[i]           # [4, 4, 6]
            x_decoded = x_decoded.permute(2, 0, 1)   # [6, 4, 4]
            x_decoded = x_decoded.reshape(-1)        # [96]

            arch_str = decoded_x_to_nas201_arch(
                x_decoded
            )

            acc, info = query_nas201_accuracy(
                api=api,
                arch_str=arch_str,
                dataset_name=args.dataset_name,
                hp=args.nas_hp,
                metric=args.nas_metric
            )

            if acc is None:
                continue
            else: 
                acc = float(acc) / 100.0

            new_archs.append(arch_str)
            new_accs.append(acc)
            new_infos.append(info)

        if len(new_archs) == 0:
            print("Nessuna architettura valida generata dal flow.")
            break

        new_accs_tensor = torch.tensor(new_accs).float()

        # statistiche
        mean_acc = new_accs_tensor.mean().item()

        if len(new_accs_tensor) > 1:
            std_acc = new_accs_tensor.std(unbiased=False).item()
        else:
            std_acc = 0.0

        min_acc = new_accs_tensor.min().item()
        max_acc = new_accs_tensor.max().item()

        print("\nGenerated NAS201 architectures from FLOW:")
        print(f"valid archs = {len(new_archs)} / {recon_probs_new.shape[0]}")
        print(f"mean acc    = {mean_acc:.4f}")
        print(f"std acc     = {std_acc:.4f}")
        print(f"min acc     = {min_acc:.4f}")
        print(f"max acc     = {max_acc:.4f}")

        history["epoch"].append(outer_epoch)
        history["mean_acc"].append(mean_acc)
        history["std_acc"].append(std_acc)
        history["min_acc"].append(min_acc)
        history["max_acc"].append(max_acc)

        #next population
        df_next_population = pd.DataFrame({
            "arch": new_archs,
            "acc": new_accs
        })

        X_next = []
        y_next = []

        for _, row in df_next_population.iterrows():

            arch_str = row["arch"]
            acc = float(row["acc"])

            x = arch_to_tensor(arch_str)
            x = torch.from_numpy(x).float().view(-1)

            X_next.append(x)
            y_next.append(acc)

        X_next = torch.stack(X_next)
        y_next = torch.tensor(y_next).float().view(-1)

        pop_size = len(df_next_population)
        unique_archs = df_next_population["arch"].nunique()

        history["population_size"].append(pop_size)

        print("\nNext population:")
        print(f"use_top_mutations = {args.use_top_mutations}")
        print(f"population size   = {pop_size}")
        print(f"unique archs      = {unique_archs}")
        print(f"duplicates        = {pop_size - unique_archs}")
        print(f"mean acc          = {df_next_population['acc'].mean():.4f}")
        print(f"max acc           = {df_next_population['acc'].max():.4f}")

        train_loader = DataLoader(
            TensorDataset(X_next, y_next),
            batch_size=args.batch_size,
            shuffle=True
        )

    print("\n TRAINING FINISHED ")

    return history, model_VAE, flow

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training FlowNAS on NAS-Bench-201"
    )
    parser.add_argument("--outer_epochs", type=int, default=20)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=16)

    parser.add_argument("--vae_epochs", type=int, default=200)
    parser.add_argument("--pretrain_vae_epochs", type=int, default=300)
    parser.add_argument("--pretrain_fraction", type=float, default=1.0)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)


    parser.add_argument("--flow_epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)


    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lambda_acc", type=float, default=5.0)

    # random perturbations
    parser.add_argument("--use_top_mutations", action="store_true")
    parser.add_argument("--elite_fraction", type=float, default=0.1)
    parser.add_argument("--mutation_fraction", type=float, default=0.2)
    parser.add_argument("--mutation_k", type=int, default=1)

    # NAS DATASET
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--nas_hp", type=str, default="200")
    parser.add_argument("--nas_metric", type=str, default="test-accuracy")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    print("\n CONFIG")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    history, model_VAE, flow = run_training(args)
