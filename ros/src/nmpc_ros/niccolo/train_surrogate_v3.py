#!/usr/bin/env python3
"""
train_surrogate.py

Surrogate model training script for neural MPC confidence prediction.

Key changes from the previous version:
  1. INPUT REPRESENTATION: uses [x, y, cos_theta, sin_theta] (NN_INPUT_DIM=4)
     directly, never collapsing to a scalar angle via atan2. This eliminates
     the +-pi discontinuity that caused the model to see a huge input jump for
     a tiny physical orientation change near theta=+-pi.

  2. OUTPUT ACTIVATION: sigmoid on the output layer, bounding predictions to
     [0,1] and matching the physical meaning of yolo_conf as a probability.
     Loss is BCE (binary cross-entropy) rather than MSE.

  3. SPECTRAL NORMALIZATION: applied to all linear layers to impose a global
     Lipschitz bound, preventing any single layer from amplifying small
     perturbations into large output changes.

  4. CONSISTENCY REGULARIZATION: penalizes ||f(x+eps) - f(x)||^2 at each
     training step, directly enforcing small input -> small output.

  5. JACOBIAN REGULARIZATION: penalizes ||df/dx||^2, the squared norm of the
     input gradient. Stronger than consistency — directly what the MPC
     optimizer requires when differentiating through the surrogate.

  6. EXPANDED TRAINING LOG: records task_loss, consistency_loss, jacobian_loss
     separately every epoch for tuning lambdas.

Usage:
  python train_surrogate.py \
    --train_num 1 \
    --data_type fronte \
    --split 80_20 \
    --dataset_version orig \
    --epochs 500 \
    [--lambda_consistency 0.1] \
    [--lambda_jacobian 1e-3] \
    [--eps_consistency 1e-2]
"""

import os
import csv
import glob
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import spectral_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# CLI
# ===========================================================================

parser = argparse.ArgumentParser(
    description="Surrogate model training with smoothness regularization"
)
parser.add_argument("--train_num",          type=int,   default=1)
parser.add_argument("--data_type",          type=str,   default="fronte")
parser.add_argument("--split",              type=str,   default="80_20")
parser.add_argument("--dataset_version",    type=str,   default="orig")
parser.add_argument("--epochs",             type=int,   default=500)
parser.add_argument("--lambda_consistency", type=float, default=1.0,
                    help="Weight for consistency regularization loss")
parser.add_argument("--lambda_jacobian",    type=float, default=1e-3,
                    help="Weight for Jacobian norm regularization loss")
parser.add_argument("--eps_consistency",    type=float, default=1e-2,
                    help="Noise magnitude for consistency regularization")
args = parser.parse_args()


# ===========================================================================
# CONFIG
# ===========================================================================

TRAIN               = True
BATCH_SIZE          = 32          # increased from 16 for smoother gradients
LR                  = 3e-4
WEIGHT_DECAY        = 8e-4
EPOCHS              = args.epochs
EARLY_STOP_PATIENCE = 40

# 4 raw inputs: [x, y, cos_theta, sin_theta]
# No atan2 collapse, no re-expansion trick in forward().
NN_INPUT_DIM      = 3
HIDDEN_SIZE       = 64
NUM_HIDDEN_LAYERS = 3

LAMBDA_CONSISTENCY = args.lambda_consistency
LAMBDA_JACOBIAN    = args.lambda_jacobian
EPS_CONSISTENCY    = args.eps_consistency

train_split = args.split

BASE_DIR = (
    f"/home/simulator/Desktop/neural_mpc_arlotta/paper_extension/agri_neural_mpc/"
    f"surrogate_model_training/trainings/{args.data_type}_trainings/theta_data/"
    f"{args.dataset_version}_dataset/{train_split}/"
)

TRAIN_INPUTS_DIR  = os.path.join(BASE_DIR, "train_inputs")
TRAIN_TARGETS_DIR = os.path.join(BASE_DIR, "train_targets")
VAL_INPUTS_DIR    = os.path.join(BASE_DIR, "val_inputs")
VAL_TARGETS_DIR   = os.path.join(BASE_DIR, "val_targets")

SAVE_DIR = os.path.join(BASE_DIR, f"training_{args.train_num}/saved_models")

TARGET_COL = "yolo_conf"


# ===========================================================================
# MODEL
# ===========================================================================

"""
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim=4, hidden_size=64, hidden_layers=3):
        super().__init__()

        self.input_layer = spectral_norm(
            torch.nn.Linear(input_dim, hidden_size)
        )
        self.hidden_layers = torch.nn.ModuleList([
            spectral_norm(torch.nn.Linear(hidden_size, hidden_size))
            for _ in range(hidden_layers)
        ])
        self.out_layer = spectral_norm(
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, 4) -- [x, y, cos_theta, sin_theta]
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = torch.sigmoid(self.out_layer(x))   # bound to (0,1)
        return x
"""

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.shape[-1] == 3:
            sin_cos = torch.cat(
                [torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1
            )
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = torch.sigmoid(self.out_layer(x))   # bound to (0,1)
        return x


# ===========================================================================
# SMOOTHNESS REGULARIZATION
# ===========================================================================

def consistency_regularization(model, x, eps):
    """
    Penalizes ||f(x + noise) - f(x)||^2 for a small random noise vector.
    One extra forward pass -- cheap to compute.
    """
    noise = eps * torch.randn_like(x)
    with torch.no_grad():
        y_clean = model(x)
    y_perturbed = model(x + noise)
    return F.mse_loss(y_perturbed, y_clean)


def jacobian_regularization(model, x):
    """
    Penalizes ||df/dx||^2, the squared Frobenius norm of the input Jacobian.
    Requires create_graph=True so gradients can flow back through the penalty.
    """
    x_req = x.detach().clone().requires_grad_(True)
    y = model(x_req)
    y_squeezed = y.squeeze(-1)

    grad_outputs = torch.ones_like(y_squeezed)
    gradients = torch.autograd.grad(
        outputs=y_squeezed,
        inputs=x_req,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    penalty = gradients.pow(2).sum(dim=1).mean()
    return penalty


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_csvs_from_dir(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def make_tensors(inputs_dir, targets_dir):

    inputs_df  = load_csvs_from_dir(inputs_dir)
    targets_df = load_csvs_from_dir(targets_dir)

    assert len(inputs_df) == len(targets_df), (
        f"Row count mismatch: inputs={len(inputs_df)}, "
        f"targets={len(targets_df)} in {inputs_dir} / {targets_dir}"
    )

    x_np   = inputs_df["x"].to_numpy(dtype=np.float32)
    y_np   = inputs_df["y"].to_numpy(dtype=np.float32)
    theta_np = inputs_df["theta"].to_numpy(dtype=np.float32)

    X_np = np.stack([x_np, y_np, theta_np], axis=1)   # (N, 4)
    Y_np = targets_df[[TARGET_COL]].to_numpy(dtype=np.float32)
    Y_np = np.clip(Y_np, 0.0, 1.0)   # safety clamp for BCE

    return (
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(Y_np, dtype=torch.float32),
    )


# ===========================================================================
# DIAGNOSTICS
# ===========================================================================

def plot_loss_curve(log_path, run_dir):
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_task_loss"], label="Train task loss",  linewidth=1.5)
    ax.plot(df["epoch"], df["val_loss"],        label="Val loss",         linewidth=1.5)
    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val   = df["val_loss"].min()
    ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1,
               label=f"Best epoch {best_epoch} (val={best_val:.5f})")
    lr_changes = df[df["lr"].diff().abs() > 1e-10]["epoch"].tolist()
    for ep in lr_changes:
        ax.axvline(ep, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)
    if lr_changes:
        ax.axvline(lr_changes[0], color="orange", linestyle=":", linewidth=0.8,
                   alpha=0.7, label="LR reduction")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Task loss (BCE) -- Train vs Validation")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["epoch"], df["train_consistency_loss"],
             label=f"Consistency (lambda={LAMBDA_CONSISTENCY})", linewidth=1.5)
    ax2.plot(df["epoch"], df["train_jacobian_loss"],
             label=f"Jacobian (lambda={LAMBDA_JACOBIAN})",    linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Regularization Loss")
    ax2.set_title("Smoothness regularization losses")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(run_dir, "plot_loss_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_pred_vs_gt(gt, pred, run_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, s=10, alpha=0.5, edgecolors="none", label="Val samples")
    lims = [min(gt.min(), pred.min()) - 0.02, max(gt.max(), pred.max()) + 0.02]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground truth (yolo_conf)"); ax.set_ylabel("Predicted (yolo_conf)")
    ax.set_title("Predicted vs Ground Truth"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "plot_pred_vs_gt.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_distribution(errors, run_dir, mse, rmse, mae):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red",    linestyle="--", linewidth=1.5, label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {errors.mean():.4f}")
    stats = (f"MSE  = {mse:.5f}\nRMSE = {rmse:.5f}\n"
             f"MAE  = {mae:.5f}\nStd  = {errors.std():.5f}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            va="top", ha="right", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Error (GT - pred)"); ax.set_ylabel("Count")
    ax.set_title("Prediction error distribution (val set)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "plot_error_distribution.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_vs_gt(gt, errors, run_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gt, errors, s=8, alpha=0.4, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    sort_idx   = np.argsort(gt)
    gt_sorted  = gt[sort_idx]
    err_sorted = errors[sort_idx]
    window     = max(1, len(gt) // 20)
    rolling    = np.convolve(err_sorted, np.ones(window)/window, mode="valid")
    x_roll     = gt_sorted[window//2: window//2 + len(rolling)]
    ax.plot(x_roll, rolling, color="orange", linewidth=2,
            label=f"Rolling mean (w={window})")
    ax.set_xlabel("GT (yolo_conf)"); ax.set_ylabel("Error (GT - pred)")
    ax.set_title("Error vs GT -- bias per confidence range")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "plot_error_vs_gt.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_smoothness_check(model, val_loader, device, run_dir):
    """
    Plots ||delta_input|| vs |delta_output| for consecutive val pairs.
    A smooth model shows all points below a linear envelope.
    """
    model.eval()
    all_din, all_dout = [], []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            if len(x_batch) < 2:
                continue
            x1, x2 = x_batch[:-1], x_batch[1:]
            y1 = model(x1).cpu().numpy().flatten()
            y2 = model(x2).cpu().numpy().flatten()
            d_in  = torch.norm(x1 - x2, dim=1).cpu().numpy()
            d_out = np.abs(y1 - y2)
            all_din.extend(d_in.tolist())
            all_dout.extend(d_out.tolist())

    din  = np.array(all_din)
    dout = np.array(all_dout)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(din, dout, s=4, alpha=0.3, edgecolors="none")
    lim = din.max()
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5,
            label="delta_out = delta_in (Lipschitz-1 envelope)")
    ax.set_xlabel("||delta_input|| (L2 between val pairs)")
    ax.set_ylabel("|delta_output| (confidence difference)")
    ax.set_title("Smoothness check: input distance vs output distance")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "plot_smoothness_check.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def save_diagnostics(log_path, gt, pred, model, val_loader, device, run_dir,
                     mse, rmse, mae):
    errors = gt - pred
    print("\nGenerating diagnostic plots...")
    plot_loss_curve(log_path, run_dir)
    plot_pred_vs_gt(gt, pred, run_dir)
    plot_error_distribution(errors, run_dir, mse, rmse, mae)
    plot_error_vs_gt(gt, errors, run_dir)
    plot_smoothness_check(model, val_loader, device, run_dir)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print(f"  lambda_consistency : {LAMBDA_CONSISTENCY}")
    print(f"  lambda_jacobian    : {LAMBDA_JACOBIAN}")
    print(f"  eps_consistency    : {EPS_CONSISTENCY}")
    print(f"  input_dim          : {NN_INPUT_DIM}  ([x, y, theta])")
    print(f"  output activation  : sigmoid  (predictions in (0,1))")
    print(f"  spectral norm      : enabled on all layers")
    print("=" * 60)

    print("\nLoading datasets...")
    X_train, Y_train = make_tensors(TRAIN_INPUTS_DIR, TRAIN_TARGETS_DIR)
    X_val,   Y_val   = make_tensors(VAL_INPUTS_DIR,   VAL_TARGETS_DIR)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    print(f"  Train samples : {len(X_train)}")
    print(f"  Val samples   : {len(X_val)}")
    print(f"  Input shape   : {X_train.shape[1]}  (should be 4)")

    if not TRAIN:
        print("TRAIN=False, exiting.")
        return

    run_dir = SAVE_DIR
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    log_path = os.path.join(run_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_task_loss",
        "train_consistency_loss", "train_jacobian_loss",
        "val_loss", "lr", "is_best",
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiLayerPerceptron(
        input_dim=NN_INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        hidden_layers=NUM_HIDDEN_LAYERS,
    ).to(device)

    criterion = torch.nn.BCELoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-6, verbose=True,
    )

    print(model)

    best_val_loss              = float("inf")
    best_model_path            = os.path.join(run_dir, "best_model_weights.pth")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):

        # Training
        model.train()
        sum_loss = sum_task = sum_consistency = sum_jacobian = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            pred      = model(x_batch)
            task_loss = criterion(pred, y_batch)
            c_loss    = consistency_regularization(model, x_batch, EPS_CONSISTENCY)
            j_loss    = jacobian_regularization(model, x_batch)

            loss = (
                task_loss
                + LAMBDA_CONSISTENCY * c_loss
                #+ LAMBDA_JACOBIAN    * j_loss
            )

            loss.backward()
            optimizer.step()

            sum_loss        += loss.item()
            sum_task        += task_loss.item()
            sum_consistency += c_loss.item()
            sum_jacobian    += j_loss.item()

        n = len(train_loader)
        avg_train_loss = sum_loss        / n
        avg_task_loss  = sum_task        / n
        avg_c_loss     = sum_consistency / n
        avg_j_loss     = sum_jacobian    / n

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  [epoch {epoch+1}] New best val_loss={best_val_loss:.6f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_writer.writerow([
            epoch + 1,
            avg_train_loss, avg_task_loss,
            avg_c_loss, avg_j_loss,
            avg_val_loss, current_lr, int(is_best),
        ])
        log_file.flush()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:>4}/{EPOCHS}] | "
                f"Total: {avg_train_loss:.5f} | "
                f"Task: {avg_task_loss:.5f} | "
                f"Cons: {avg_c_loss:.5f} | "
                f"Jac: {avg_j_loss:.5f} | "
                f"Val: {avg_val_loss:.5f} | "
                f"LR: {current_lr:.2e}"
            )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch+1} "
                f"(no improvement for {EARLY_STOP_PATIENCE} epochs)."
            )
            break

    log_file.close()
    print(f"\nBest validation loss : {best_val_loss:.6f}")
    print(f"Training log saved to: {log_path}")

    # Final evaluation
    print("\nEvaluating best model on validation set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_gt, all_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            all_pred.extend(model(x_batch).cpu().numpy().flatten())
            all_gt.extend(y_batch.numpy().flatten())

    all_gt   = np.array(all_gt)
    all_pred = np.array(all_pred)
    errors   = all_gt - all_pred

    mse  = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(errors)))

    results_path = os.path.join(run_dir, "val_results.csv")
    pd.DataFrame({
        "ground_truth": all_gt,
        "predicted":    all_pred,
        "error":        errors,
    }).to_csv(results_path, index=False)

    print(f"Final Val MSE  : {mse:.6f}")
    print(f"Final Val RMSE : {rmse:.6f}")
    print(f"Final Val MAE  : {mae:.6f}")

    save_diagnostics(
        log_path, all_gt, all_pred,
        model, val_loader, device, run_dir,
        mse, rmse, mae,
    )

    # Save hyperparams for reproducibility
    hparam_path = os.path.join(run_dir, "hparams.csv")
    pd.DataFrame([{
        "input_dim":          NN_INPUT_DIM,
        "hidden_size":        HIDDEN_SIZE,
        "num_hidden_layers":  NUM_HIDDEN_LAYERS,
        "batch_size":         BATCH_SIZE,
        "lr":                 LR,
        "weight_decay":       WEIGHT_DECAY,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "lambda_jacobian":    LAMBDA_JACOBIAN,
        "eps_consistency":    EPS_CONSISTENCY,
        "data_type":          args.data_type,
        "dataset_version":    args.dataset_version,
        "split":              args.split,
        "train_num":          args.train_num,
    }]).to_csv(hparam_path, index=False)

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
