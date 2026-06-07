import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset

from models.flow import FlowNet
from models.nas201_models import VAE_dist, vae_accuracy_loss,vae_accuracy_loss_ws
from models.nas301_models import  VAE_nas301,vae_accuracy_loss_nas301

import os
import tarfile
from nats_bench import create
import torch.nn.functional as F
from ws_universale.supernet import Supernet
from ws_universale.nb201 import nasbench201_strings_to_networkdags
import torch.nn as nn


from dataset_loader import (
    NASDatasetFactory,
    load_nas201_api,
    arch_to_tensor, tensor_to_genotype,
    load_nas301_performance_model, genotype_to_tensor
)

from utils_functions.utils import (
    build_accuracy_pairs,
    set_seed,
    generate_archs,
    decoded_x_to_nas201_arch,
    query_nas201_accuracy,
    query_nas301_accuracy,
    decode_population_nas301,
    build_next_population,
    get_cifar10_loaders,
    random_nas201_arch
)

def run_training(args):

    DEVICE = torch.device(args.device)
    set_seed(args.seed)

    benchmark_name = args.benchmark_name.upper()
    api = None
    performance_model = None
    weight_sharing = args.weight_sharing

    if benchmark_name == "NAS201":
        #dovrebbe essere if weight_sharing carico solo il dataset, else carico l'api 
        if weight_sharing:
            supernet = Supernet();
            if args.dataset_name == "cifar10":
                train_dataset_loader, val_dataset_loader = get_cifar10_loaders(batch_size=256)

            loss_fn = vae_accuracy_loss_ws
        else: 
            loss_fn = vae_accuracy_loss

        #else:
        if hasattr(args, "api") and args.api is not None:
            api = args.api
        else:
            api = load_nas201_api()
        set_seed(args.seed)

        model_VAE = VAE_dist(
            INPUT_DIM=80,
            LATENT_DIM=args.latent_dim,
            output_shape=(4, 4, 5)
        ).to(DEVICE)

    elif benchmark_name == "NAS301":

        if getattr(args, "performance_model", None) is not None:
            performance_model = args.performance_model
        else:
            performance_model = load_nas301_performance_model()
            print(performance_model)

        set_seed(args.seed)
        model_VAE = VAE_nas301(
            INPUT_DIM=504,
            LATENT_DIM=args.latent_dim,
            output_shape=(2, 7, 6, 6)
        ).to(DEVICE)

        loss_fn = vae_accuracy_loss_nas301

    else:
        raise ValueError(f"Benchmark non supportato: {args.benchmark_name}")

    set_seed(args.seed)
    flow = FlowNet(dim=args.latent_dim).to(DEVICE)
    
    #Dataset mi serve solo nel caso in cui non ho weight sharing:
    if args.train_dataset is None and not weight_sharing:
        print("Dataset not available, importing...")

        if args.benchmark_name.upper() == "NAS201":

            train_dataset,test_dataset,train_loader,test_loader = NASDatasetFactory.create(
                benchmark_name="NAS201",
                api=api,
                dataset_name=args.dataset_name,
                metric=args.nas_metric,
                hp=args.nas_hp,
                flatten=True,
                normalize_y=True,
            )

            performance_model = None
            train_size = len(train_dataset)
            test_size = len(test_dataset)

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
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            generator = torch.Generator().manual_seed(args.seed)

            train_dataset, test_dataset = random_split(
                dataset,
                [train_size, test_size],
                generator=generator
            )
            print("Numero architetture:", len(dataset))  #spostato non so perchè stava sotto


        else:
            raise ValueError(f"Benchmark non supportato: {args.benchmark_name}")

        
    elif not weight_sharing and args.train_dataset is not None: 
        print("Dataset available")
        train_size = len(args.train_dataset)
        test_size = len(args.test_dataset)
        train_dataset = args.train_dataset
        test_dataset = args.test_dataset

    # training
    #1. VAE pretrain 

    print("\nPRETRAIN VAE")

    if weight_sharing:
        # Generazione di architetture casuali senza accuracy
        X_pretrain = []

        for _ in range(args.n_samples):
            arch_str = random_nas201_arch()
            x = arch_to_tensor(arch_str)
            X_pretrain.append(
                torch.from_numpy(x).float().view(-1)
            )

        X_pretrain = torch.stack(X_pretrain)

        pretrain_dataset = TensorDataset(
            X_pretrain,
            torch.rand(X_pretrain.size(0)),
        )

        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=args.pretrain_batch_size,
            shuffle=True
        )

    else:
        n_pretrain = int(
            train_size * args.pretrain_fraction
        )
        generator = torch.Generator().manual_seed(
            args.seed
        )

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

    if benchmark_name == "NAS301":
        pretrain_and_freeze_vae(
            model_VAE=model_VAE,
            pretrain_loader=pretrain_loader,
            loss_fn=loss_fn,
            vae_epochs=args.pretrain_vae_epochs,
            beta=args.beta,
            lambda_acc=args.lambda_acc,
            DEVICE=DEVICE,
            pos_weight_value=args.pos_weight_value
        )

    if benchmark_name == "NAS201":
        if weight_sharing:
            # Loss solo ricostruttiva: ignoriamo l'accuracy (lambda_acc=0)
            pretrain_and_freeze_vae(
                model_VAE=model_VAE,
                pretrain_loader=pretrain_loader,
                loss_fn=loss_fn,          # vae_accuracy_loss_ws
                vae_epochs=args.pretrain_vae_epochs,
                beta=args.beta,
                lambda_acc=0.0,           # disabilita il termine accuracy
                DEVICE=DEVICE
            )
        else:
            pretrain_and_freeze_vae(
                model_VAE=model_VAE,
                pretrain_loader=pretrain_loader,
                loss_fn=loss_fn,          # vae_accuracy_loss
                vae_epochs=args.pretrain_vae_epochs,
                beta=args.beta,
                DEVICE=DEVICE
            )

    converter = (
        tensor_to_genotype
        if benchmark_name == "NAS301"
        else decoded_x_to_nas201_arch
    )

    source = X_pretrain if weight_sharing else train_dataset

    current_pop = generate_archs(
        data=source,
        converter=converter,
        N=args.N,
        seed=args.seed
    )

    history = {
        "epoch": [],
        "mean_acc": [],
        "std_acc": [],
        "min_acc": [],
        "max_acc": [],
        "population_size": []
    }
    df_current_population = None

    for outer_epoch in range(args.outer_epochs):

        print(
            f"\n OUTER EPOCH "
            f"{outer_epoch + 1}/{args.outer_epochs} =========="
        )
        if benchmark_name == "NAS301":
            evaluated_archs = []
            new_accs = []

            for genotype in current_pop:
                acc, info = query_nas301_accuracy(
                    performance_model,
                    genotype
                )

                if acc is not None:
                    evaluated_archs.append(genotype)
                    new_accs.append(float(acc))

        elif benchmark_name == "NAS201":
            evaluated_archs = current_pop

            if weight_sharing:
                print("Evaluating networks...")

                network_dags = nasbench201_strings_to_networkdags(
                    current_pop
                )

                raw_accs = supernet.eval_subnets(
                    networks=network_dags,
                    train_loader=train_dataset_loader,
                    eval_loader=val_dataset_loader,
                    device=DEVICE,
                    bn_batches=20,
                    epochs=20,
                    calibrate=True,
                    M=4,
                    criterion=nn.CrossEntropyLoss(
                        label_smoothing=0.1
                    ),
                    use_label_smoothing=True,
                    scheduler_factory=lambda opt:
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt,
                            T_max=200,
                            eta_min=1e-4
                        ),
                    optimizer_factory=lambda params:
                        torch.optim.SGD(
                            params,
                            lr=0.025,
                            momentum=0.9,
                            weight_decay=3e-4,
                            nesterov=True
                        ),
                )

                new_accs = list(map(float, raw_accs))

            else:
                new_accs = []

                for arch_str in current_pop:
                    acc, info = query_nas201_accuracy(
                        api=api,
                        arch_str=arch_str,
                        dataset_name=args.dataset_name,
                        hp=args.nas_hp,
                        metric=args.nas_metric
                    )

                    new_accs.append(
                        float(acc) / 100.0
                        if acc is not None
                        else 0.0
                    )

        else:
            raise ValueError(
                f"Benchmark non supportato: {benchmark_name}"
            )

        if not new_accs:
            print("Nessuna architettura valutata correttamente.")
            break

        new_accs_tensor = torch.tensor(
            new_accs,
            dtype=torch.float32
        )

        mean_acc = new_accs_tensor.mean().item()
        std_acc = new_accs_tensor.std(unbiased=False).item()
        min_acc = new_accs_tensor.min().item()
        max_acc = new_accs_tensor.max().item()

        print(f"\nEvaluated {benchmark_name} architectures:")
        print(f"valid archs = {len(evaluated_archs)} / {len(current_pop)}")
        print(f"mean acc    = {mean_acc:.4f}")
        print(f"std acc     = {std_acc:.4f}")
        print(f"min acc     = {min_acc:.4f}")
        print(f"max acc     = {max_acc:.4f}")

        history["epoch"].append(outer_epoch)
        history["mean_acc"].append(mean_acc)
        history["std_acc"].append(std_acc)
        history["min_acc"].append(min_acc)
        history["max_acc"].append(max_acc)
        history["population_size"].append(len(evaluated_archs))

        converter = (
            genotype_to_tensor
            if benchmark_name == "NAS301"
            else arch_to_tensor
        )

        X_current = torch.stack([
            torch.from_numpy(converter(arch)).float().flatten()
            for arch in evaluated_archs
        ])

        y_current = torch.tensor(new_accs, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_current, y_current),
            batch_size=args.batch_size,
            shuffle=True
        )

        result =  train_one_epoch(
                flow = flow,
                model_VAE = model_VAE,
                train_loader = train_loader,
                flow_epochs=100,
                alpha=0.5,
                DEVICE=DEVICE
            )

        if result is None:
            print("Training interrotto: nessuna coppia valida trovata.")
            break
        
        z_new, z_all, y_all = result
        model_VAE.eval()

        with torch.no_grad():
            decoded = model_VAE.decode(
                z_new.to(DEVICE).float()
            )

        decoded_archs = decoded[-1].cpu()

        if benchmark_name == "NAS301":
            current_pop, df_next_population = build_next_population(
                decoded_archs=decoded_archs,
                current_pop=current_pop,
                current_accs=new_accs,
                converter=lambda x: tensor_to_genotype(x.flatten()),
                elite_fraction=args.elite_fraction,
                max_population_size=args.N,
                key_fn=str,
            )
        elif benchmark_name == "NAS201":
            current_pop, df_next_population = build_next_population(
                decoded_archs=decoded_archs,
                current_pop=current_pop,
                current_accs=new_accs,
                converter=lambda x: decoded_x_to_nas201_arch(
                    x.permute(2, 0, 1).flatten()
                ),
                elite_fraction=args.elite_fraction,
                max_population_size=args.N,
                key_fn=str,
            )

            current_pop = df_next_population["arch"].tolist()

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
    parser.add_argument("--WS", type=str, default="False")



    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    print("\n CONFIG")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    history, model_VAE, flow = run_training(args)


def train_one_epoch(
    flow,
    model_VAE,
    train_loader,
    flow_epochs=100,
    alpha=0.5,
    DEVICE="cpu"
):
    """
    1. Extract embeddings of train_loader using model_VAE
    2. Build improving accuracy pairs
    3. Train the Flow
    4. Generate new architectures using Flow
    """

    model_VAE = model_VAE.to(DEVICE)
    flow = flow.to(DEVICE)

    # 1. Embeddings extraction con VAE già allenato
    model_VAE.eval()

    z_all = []
    y_all = []

    with torch.no_grad():

        for x, y in train_loader:
            x = x.to(DEVICE).float()
            y = y.float().view(-1)
            mu, logvar = model_VAE.encode(x)

            z_all.append(mu.cpu())
            y_all.append(y.cpu())

    z_all = torch.cat(z_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    print("z_all shape:", z_all.shape)
    print("y_all shape:", y_all.shape)

    # 2. Pair generation
    pairs_x, pairs_target = build_accuracy_pairs(
        X=z_all,
        y=y_all,
        K=50,
        min_delta_acc=0.01,
        seed=42
    )

    if len(pairs_x) == 0:
        print("Nessuna coppia trovata: prova ad aumentare K o abbassare min_delta_acc.")
        return None
    print("pairs_x shape:", pairs_x.shape)
    print("pairs_target shape:", pairs_target.shape)

    pairs_dataset = TensorDataset(
        pairs_x,
        pairs_target
    )
    pairs_loader = DataLoader(
        pairs_dataset,
        batch_size=64,
        shuffle=True
    )

    # 3. Training flow matching
    flow_optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=1e-3
    )
    flow.train()

    for epoch in range(flow_epochs):
        total_flow_loss = 0.0

        for z_start, direction_target in pairs_loader:
            z_start = z_start.to(DEVICE).float()
            direction_target = direction_target.to(DEVICE).float()
            pred_direction = flow(z_start)

            loss = F.mse_loss(
                pred_direction,
                direction_target
            )

            flow_optimizer.zero_grad()
            loss.backward()
            flow_optimizer.step()
            total_flow_loss += loss.item()

    # 4. Generate new architectures from flow
    flow.eval()

    with torch.no_grad():

        z_start = z_all.to(DEVICE).float()

        direction = flow(z_start)

        z_new = z_start + alpha * direction

    print("z_new shape:", z_new.shape)

    return z_new, z_all, y_all



def pretrain_and_freeze_vae(
    model_VAE,
    pretrain_loader,
    loss_fn,                     
    beta=0.0,
    lambda_acc=1.0,
    vae_epochs=300,
    DEVICE="cpu",
    early_stop=True,
    patience=20,
    min_delta=1e-5,
    loss_threshold=1e-4,
    lr=1e-3,
    **loss_kwargs                
):
    """
    Pretrains a VAE and freezes it.

     Input:
    model_VAE : VAE model to pretrain.
    pretrain_loader : `x` is the input and `y`
        is the target accuracy.
    loss_fn : callable Loss function used for VAE pretraining.
    beta : Weight of the KL-divergence term.
    lambda_acc : Weight of the accuracy-prediction loss.
    vae_epochs : Maximum number of pretraining epochs.
    DEVICE : Training device.
    early_stop :  Whether to stop training early based on loss improvement.
    patience : Number of epochs tolerated without improvement.
    min_delta :Minimum loss improvement required to reset patience.
    loss_threshold : Loss value below which training stops immediately.
    lr : Adam optimizer learning rate.
    **loss_kwargs
        Additional arguments passed to `loss_fn`.

     Output:
    The pretrained and frozen VAE.
    """

    model_VAE = model_VAE.to(DEVICE)
    optimizer = torch.optim.Adam(
        model_VAE.parameters(),
        lr=lr
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(vae_epochs):

        model_VAE.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_acc_loss = 0.0

        for x, y in pretrain_loader:

            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1)

            recon_logits, recon_probs, mu, logvar, acc_pred = model_VAE(x)
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
                **loss_kwargs
            )

            if len(result) == 4:
                loss, recon_loss, kl, acc_loss = result
            elif len(result) == 3:
                loss, recon_loss, kl = result
                acc_loss = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            total_acc_loss += acc_loss.item()

        avg_loss = total_loss / len(pretrain_loader)
        avg_recon = total_recon / len(pretrain_loader)
        avg_kl = total_kl / len(pretrain_loader)
        avg_acc_loss = total_acc_loss / len(pretrain_loader)

        if epoch % 50 == 0:
            print(
                f"VAE pretrain epoch {epoch:03d} | "
                f"loss={avg_loss:.6f} | "
                f"recon={avg_recon:.6f} | "
                f"kl={avg_kl:.6f} | "
                f"acc_loss={avg_acc_loss:.6f}"
            )

        if early_stop:

            if avg_loss < loss_threshold:
                print(
                    f"Early stopping: loss below threshold "
                    f"at epoch {epoch}, loss={avg_loss:.6f}"
                )
                break
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping: patience reached "
                    f"at epoch {epoch}, best_loss={best_loss:.6f}"
                )
                break

    model_VAE.eval()

    for p in model_VAE.parameters():
        p.requires_grad = False

    print("VAE pretrained and frozen.")

    return model_VAE
    