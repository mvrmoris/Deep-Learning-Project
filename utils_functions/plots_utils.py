import matplotlib.ticker as ticker
from scipy.stats import norm
import numpy as np 
import torch
import matplotlib.pyplot as plt

def compare_accuracy_distributions(y_all, new_accs, title="NAS201 Accuracy: Initial vs Generated",path = None):

    if isinstance(y_all, torch.Tensor):
        y_init = y_all.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        y_init = np.array(y_all, dtype=np.float32).reshape(-1)

    y_gen = np.array([a for a in new_accs if a is not None], dtype=np.float32)

    if len(y_gen) == 0:
        raise ValueError("new_accs contiene solo valori None.")

    if y_init.max() <= 1.5:
        y_init = y_init * 100.0
    if y_gen.max() <= 1.5:
        y_gen = y_gen * 100.0

    mu_i,  std_i,  var_i  = y_init.mean(), y_init.std(),  y_init.var()
    mu_g,  std_g,  var_g  = y_gen.mean(),  y_gen.std(),   y_gen.var()

    print(f"INITIAL    — n={len(y_init):4d}  mean={mu_i:.3f}  var={var_i:.3f}  std={std_i:.3f}  min={y_init.min():.3f}  max={y_init.max():.3f}")
    print(f"GENERATED  — n={len(y_gen):4d}  mean={mu_g:.3f}  var={var_g:.3f}  std={std_g:.3f}  min={y_gen.min():.3f}  max={y_gen.max():.3f}")
    print(f"Δ mean = {mu_g - mu_i:+.3f}")

    margin = max(std_i, std_g) * 4.0
    xs     = np.linspace(min(mu_i, mu_g) - margin, max(mu_i, mu_g) + margin, 800)
    pdf_i  = norm.pdf(xs, mu_i, std_i)
    pdf_g  = norm.pdf(xs, mu_g, std_g)

    C_INIT, C_GEN, BG = "#4A90D9", "#C0503A", "#F7F7F5"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#CCCCCC")
    ax.spines["bottom"].set_edgecolor("#CCCCCC")

    ax.fill_between(xs, pdf_i, alpha=0.18, color=C_INIT)
    ax.fill_between(xs, pdf_g, alpha=0.18, color=C_GEN)
    ax.plot(xs, pdf_i, color=C_INIT, linewidth=2.2,
            label=f"Initial   (μ={mu_i:.2f},  σ²={var_i:.2f})")
    ax.plot(xs, pdf_g, color=C_GEN,  linewidth=2.2,
            label=f"Generated (μ={mu_g:.2f},  σ²={var_g:.2f})")

    ax.axvline(mu_i, color=C_INIT, linewidth=1.3, linestyle="--", alpha=0.9)
    ax.axvline(mu_g, color=C_GEN,  linewidth=1.3, linestyle="--", alpha=0.9)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Accuracy (%)", fontsize=11, labelpad=8, color="#444444")
    ax.set_ylabel("Density",      fontsize=11, labelpad=8, color="#444444")
    ax.tick_params(colors="#666666", labelsize=9.5)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0, top=max(pdf_i.max(), pdf_g.max()) * 1.18)

    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#CCCCCC",
              loc="upper left", handlelength=1.8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if path is not None:
        plt.savefig(path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()

def plot_history_gaussians(
    history,
    title="NAS201 Accuracy distributions across outer epochs",
    save_path=None,
    max_gaussians=5
):
    epochs = np.array(history["epoch"])
    means = np.array(history["mean_acc"], dtype=np.float32)
    stds = np.array(history["std_acc"], dtype=np.float32)
    selected_idx = np.arange(len(means))

    epochs_plot = epochs[selected_idx]
    means_plot = means[selected_idx]
    stds_plot = stds[selected_idx]

    means_plot = means_plot * 100.0
    stds_plot = stds_plot * 100.0
    stds_plot = np.maximum(stds_plot, 1e-6)

    x_min = np.min(means_plot - 4 * stds_plot)
    x_max = np.max(means_plot + 4 * stds_plot)
    xs = np.linspace(x_min, x_max, 1000)

    BG = "white"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#CCCCCC")
    ax.spines["bottom"].set_edgecolor("#CCCCCC")

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(means_plot)))

    max_pdf = 0.0

    for epoch, mu, std, color in zip(epochs_plot, means_plot, stds_plot, colors):
        pdf = norm.pdf(xs, mu, std)
        max_pdf = max(max_pdf, pdf.max())
        ax.fill_between(xs, pdf, alpha=0.10, color=color)
        ax.plot(
            xs,
            pdf,
            color=color,
            linewidth=2.0,
            label=f"Epoch {epoch + 1} (μ={mu:.2f}, σ={std:.2f})"
        )
        ax.axvline(
            mu,
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.75
        )

    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("Accuracy (%)", fontsize=11, labelpad=8, color="#444444")
    ax.set_ylabel("Density", fontsize=11, labelpad=8, color="#444444")
    ax.tick_params(colors="#666666", labelsize=9.5)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0, top=max_pdf * 1.18)

    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        y=0.97,
        color="#2C2C2A"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=160,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    plt.show()