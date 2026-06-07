from datasets.dataset_loader_nas301 import tensor_to_genotype, genotype_to_tensor
import torch
import nasbench301 as nb
import os


def load_nas301_performance_model():
    """Return the performance model of NAS301"""
    model_dir = os.path.join("nb_models_1.0", "xgb_v1.0")

    if os.path.exists(model_dir):
        print("Pesi NAS-Bench-301 trovati localmente.")
    else:
        print("Scaricamento dei pesi NAS-Bench-301...")
        nb.download_models(version="1.0")

    model = nb.load_ensemble(model_dir)
    print("Surrogate model NAS-Bench-301 caricato con successo.")
    return model

def decode_population_nas301(model_VAE, z_new, performance_model, DEVICE):
    """decode latent vectors and query surrogate model"""

    model_VAE.eval()
    #decode architectures 
    with torch.no_grad():
        x_new = model_VAE.decode(z_new.to(DEVICE).float())[-1].cpu()

    genotypes, accs, infos = [], [], []

    #convert into NAS301 valid genotypes and query surrogate model for accuracy
    for x in x_new:
        genotype = tensor_to_genotype(x.flatten())
        acc, info = query_nas301_accuracy(performance_model, genotype)
        if acc is not None:
            genotypes.append(genotype)
            accs.append(acc)
            infos.append(info)

    return genotypes, accs, infos


def query_nas301_accuracy(
    performance_model,
    arch,
    metric="val_accuracy"):

    raw_pred = float(performance_model.predict(
        config=arch,
        representation="genotype",
        with_noise=False,
    ))

    acc = raw_pred / 100.0 if raw_pred > 1.5 else raw_pred

    return acc, {
        "raw_prediction": raw_pred,
        "normalized_accuracy": acc,
        "metric": metric,
    }