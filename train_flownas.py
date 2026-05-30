
import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset
from latent_space.model import VAE_dist, FlowNet, VAE_nas301,vae_accuracy_loss_nas301,vae_accuracy_loss
import os
import tarfile
from nats_bench import create

from latent_space.dataset_loader import (
    NASDatasetFactory,
    load_nas201_api,
    arch_to_tensor,
    load_nas301_performance_model
)

from latent_space.utils import (
    set_seed,
    pretrain_and_freeze_vae,
    generate_archs,
    train_one_epoch,
    decoded_x_to_nas201_arch,
    query_nas201_accuracy,
    query_nas301_accuracy,
    decode_population_nas301,
    build_next_population_nas301
)


def run_training(args):

    DEVICE = torch.device(args.device)
    set_seed(args.seed)

    benchmark_name = args.benchmark_name.upper()
    api = None
    performance_model = None

    if benchmark_name == "NAS201":

        if hasattr(args, "api") and args.api is not None:
            api = args.api
        else:
            api = load_nas201_api()

        model_VAE = VAE_dist(
            INPUT_DIM=96,
            LATENT_DIM=args.latent_dim,
            output_shape=(4, 4, 6)
        ).to(DEVICE)

        loss_fn = vae_accuracy_loss

    elif benchmark_name == "NAS301":

        if getattr(args, "performance_model", None) is not None:
            performance_model = args.performance_model
        else:
            performance_model = load_nas301_performance_model()
            print(performance_model)

        model_VAE = VAE_nas301(
            INPUT_DIM=504,
            LATENT_DIM=args.latent_dim,
            output_shape=(2, 7, 6, 6)
        ).to(DEVICE)

        loss_fn = vae_accuracy_loss_nas301

    else:
        raise ValueError(f"Benchmark non supportato: {args.benchmark_name}")

    flow = FlowNet(dim=args.latent_dim).to(DEVICE)
    #Dataset
    if args.train_dataset is None:

        print("Dataset not available, importing...")

        if args.benchmark_name.upper() == "NAS201":

            dataset = NASDatasetFactory.create(
                benchmark_name="NAS201",
                api=api,
                dataset_name=args.dataset_name,
                metric=args.nas_metric,
                hp=args.nas_hp,
                flatten=True,
                normalize_y=True,
            )

            performance_model = None

        elif args.benchmark_name.upper() == "NAS301":

            dataset, performance_model = NASDatasetFactory.create(
                benchmark_name="NAS301",
                performance_model=getattr(args, "performance_model", None),
                n_samples=args.n_samples,
                flatten=True,
                normalize_y=True,
                seed=args.seed
            )

            api = None

        else:
            raise ValueError(f"Benchmark non supportato: {args.benchmark_name}")

        print("Numero architetture:", len(dataset))

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(args.seed)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )
    else: 
        print("Dataset available")
        train_size = len(args.train_dataset)
        test_size = len(args.test_dataset)
        train_dataset = args.train_dataset
        test_dataset = args.test_dataset

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

    if benchmark_name =="NAS301":
        pretrain_and_freeze_vae(
            model_VAE=model_VAE,
            pretrain_loader=pretrain_loader,
            loss_fn=vae_accuracy_loss_nas301,
            vae_epochs=args.pretrain_vae_epochs,
            beta=args.beta,
            lambda_acc=args.lambda_acc,
            DEVICE=DEVICE,
            pos_weight_value=args.pos_weight_value
        )
    elif benchmark_name == "NAS201":
        pretrain_and_freeze_vae(
            model_VAE=model_VAE,
            pretrain_loader=pretrain_loader,
            loss_fn=vae_accuracy_loss,
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

        z_new, z_all, y_all = result

        if args.benchmark_name.upper() == "NAS301":

            new_genotypes, new_accs, new_infos = decode_population_nas301(
                model_VAE=model_VAE,
                z_new=z_new,
                performance_model=performance_model,
                DEVICE=DEVICE
            )

            if len(new_genotypes) == 0:
                print("Nessuna architettura NAS301 valida generata dal flow.")
                break

            new_accs_tensor = torch.tensor(new_accs).float()

            mean_acc = new_accs_tensor.mean().item()

            if len(new_accs_tensor) > 1:
                std_acc = new_accs_tensor.std(unbiased=False).item()
            else:
                std_acc = 0.0

            min_acc = new_accs_tensor.min().item()
            max_acc = new_accs_tensor.max().item()

            print("\nGenerated NAS301 architectures from FLOW:")
            print(f"valid archs = {len(new_genotypes)} / {z_new.shape[0]}")
            print(f"mean acc    = {mean_acc:.4f}")
            print(f"std acc     = {std_acc:.4f}")
            print(f"min acc     = {min_acc:.4f}")
            print(f"max acc     = {max_acc:.4f}")

            history["epoch"].append(outer_epoch)
            history["mean_acc"].append(mean_acc)
            history["std_acc"].append(std_acc)
            history["min_acc"].append(min_acc)
            history["max_acc"].append(max_acc)

            X_next, y_next, df_next_population = build_next_population_nas301(
                new_genotypes=new_genotypes,
                new_accs=new_accs,
                train_loader=train_loader,
                elite_fraction=args.elite_fraction,
                max_population_size=args.N
            )


            train_loader = DataLoader(
                TensorDataset(X_next, y_next),
                batch_size=args.batch_size,
                shuffle=True
            )
        else:
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
            # flow generated architecture: 
            generated_df = pd.DataFrame({
                "arch": new_archs,
                "acc": new_accs,
                "source": "flow"
            })

            #initial population
            current_rows = []

            for batch_x, batch_y in train_loader:

                for x_curr, y_curr in zip(batch_x, batch_y):

                    x_curr = x_curr.float().view(-1)
                    acc_curr = float(y_curr)

                    arch_curr = decoded_x_to_nas201_arch(
                        x_curr
                    )

                    current_rows.append({
                        "arch": arch_curr,
                        "acc": acc_curr,
                        "source": "elite"
                    })

            df_current_population = pd.DataFrame(current_rows)
            df_current_population = (
                    df_current_population
                    .sort_values("acc", ascending=False)
                    .reset_index(drop=True)
                )
            #taking a certain percentage of better performing architectures from previous round
            n_elite = int(len(df_current_population) * args.elite_fraction)
            n_elite = max(0, n_elite)

            elite_df = df_current_population.head(n_elite).copy()

            #concat of elite population + flow generated
            df_next_population = pd.concat(
                [generated_df, elite_df],
                ignore_index=True
            )

            before_drop = len(df_next_population)

            #dropping duplicated architectures
            df_next_population = (
                df_next_population
                .sort_values("acc", ascending=False)
                .drop_duplicates(subset=["arch"], keep="first")
                .reset_index(drop=True)
            )

            after_drop = len(df_next_population)

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
            duplicates = pop_size - unique_archs

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

    if benchmark_name == "NAS301":
        return history, model_VAE, flow, test_dataset, performance_model

    return history, model_VAE, flow, test_dataset, api

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
    parser.add_argument("--benchmark_name", type=str, default="NAS201")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--nas_hp", type=str, default="200")
    parser.add_argument("--nas_metric", type=str, default="test-accuracy")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_dataset", type=TensorDataset, default=None)
    parser.add_argument("--test_dataset", type=TensorDataset, default=None)
    parser.add_argument("--performance_model",default=None)
    parser.add_argument("--pos_weight_value", type=float, default=5.0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    print("\n CONFIG")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    history, model_VAE, flow = run_training(args)
