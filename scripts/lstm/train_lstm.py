"""LSTM on SSTA EOFs

@Author  :   Jakob Schlör
@Time    :   2022/09/11 14:35:32
@Contact :   jakob.schloer@uni-tuebingen.de
"""

import argparse
import json
import os
# %%
import time
from importlib import reload

import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from hyblim import losses
from hyblim.data import dataloader
from hyblim.data import eof
from hyblim.model import lstm
from hyblim.utils import eval as heval

PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["WANDB__SERVICE_WAIT"] = "300"
id = int(os.environ.get("SLURM_LOCALID", 0))
device = torch.device("cuda", id) if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}", flush=True)


# Config
# ======================================================================================
def make_args(ipython=False):
    """
    python train_lstm.py -eof 20 -hist 15 -horiz 9 -cond film -condscale 4 -l 1 -chrono 5 -ep 2 -lr 0.0001 -path ./ -postfix _test

    Returns:
        _type_: _description_
    """
    if ipython:
        config = dict(
            # Data
            vars=["ssta", "ssha"],
            num_traindata=None,
            hist=12,
            train_horiz=20,
            batch_size=64,
            # Model
            layers=2,
            hidden_dim=64,
            film=True,
            members=16,
            # Training
            loss_type="crps",
            gamma=0.65,
            ssha_weight=1.0,
            epochs=2,
            init_lr=5e-3,
            min_lr=3e-7,
            dry=True,
            sweep=False,
            valmetric_every=1,
            wandb_project="FiLMLSTM",
            # Saving
            postfix="_test",
            path=PATH + "/../../models/lstm/",
            evaluate=True,
        )
    else:
        parser = argparse.ArgumentParser()
        # Data
        parser.add_argument("-v", "--vars", nargs="+", default=["ssta", "ssha"], help="Variable names used.")
        parser.add_argument("-ntrain", "--num_traindata", default=None, type=int, help="Number of training datapoints.")
        parser.add_argument(
            "-hist", "--hist", default=12, type=int, help="Number of maximum history datapoints for training."
        )
        parser.add_argument(
            "-horiz", "--train_horiz", default=20, type=int, help="Number of history datapoints for training."
        )
        parser.add_argument("-batch", "--batch_size", default=64, type=int, help="Batch size.")
        # Model
        parser.add_argument("-l", "--layers", default=1, type=int, help="Number of LSTM layers.")
        parser.add_argument("-hidden", "--hidden_dim", default=64, type=int, help="Dimension of hidden dimension.")
        parser.add_argument("-film", "--film", action="store_true", help="If set, conditioning with FiLM on month.")
        parser.add_argument("-members", "--members", default=16, type=int, help="Number of output members.")
        # Training
        parser.add_argument(
            "-loss",
            "--loss_type",
            default="crps",
            type=str,
            help="Name of loss used for training, i.e. 'weighted_mse', 'mse' ",
        )
        parser.add_argument("-gamma", "--gamma", default=0.65, type=float, help="Weighting of loss, gamma^tau.")
        parser.add_argument(
            "-wssha", "--ssha_weight", default=1.0, type=float, help="Loss weight for SSHA relative to SSTA (=1.0)."
        )
        parser.add_argument("-epochs", "--epochs", default=30, type=int, help="Number of epochs.")
        parser.add_argument("-ilr", "--init_lr", default=5e-3, type=float, help="Initial learning rate.")
        parser.add_argument(
            "-mlr", "--min_lr", default=5e-6, type=float, help="Minimum learning rate for Cosineannealing."
        )
        parser.add_argument("-wandb", "--wandb_project", default="FilMLSTM", type=str, help="Wandb project name.")
        parser.add_argument("-dry", "--dry", action="store_true", help="If set, dry run.")
        parser.add_argument(
            "-valevery", "--valmetric_every", default=1, type=int, help="Compute Niño/field metrics every N epochs."
        )
        # Save model
        parser.add_argument("-path", "--path", default=PATH + "/../../models/lstm", type=str, help="Modelpath.")
        parser.add_argument(
            "-postfix", "--postfix", default="", type=str, help="Postfix to model folder, e.g. '_id_1'."
        )
        parser.add_argument("-eval", "--evaluate", action="store_true", help="Evaluate model.")
        parser.add_argument("-seed", "--seed", default=None, type=int, help="Set seed.")

        config = vars(parser.parse_args())

    path_to_data = {
        "ssta": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        "ssha": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc",
    }
    config["datapaths"] = {}
    for var in config["vars"]:
        config["datapaths"][var] = path_to_data[var]

    config["lsm_path"] = PATH + "/../../data/land_sea_mask_common.nc"
    config["n_eof"] = [20, 10]
    # Use canonical precomputed EOFs (full training period) for standard runs; for
    # num_traindata experiments keep eof_path=None to fit on the training subsample.
    config["eof_path"] = (
        None
        if config["num_traindata"] is not None
        else PATH + "/../../data/cesm2-picontrol/pca/" + eof.eof_filename(config["vars"], config["n_eof"])
    )
    config["horiz"] = 24
    config["name"] = "FilmLSTM"
    config["chrono"] = 5
    config["slurm_id"] = os.environ.get("SLURM_JOB_ID", 0000000)

    return config


# Data config
config = make_args(ipython=False)

# Set seed
if config.get("seed") is not None:
    from hyblim.utils.seed import seed_everything

    seed_everything(config["seed"])
    print(f"Seed set to {config['seed']}", flush=True)

# %%
# Load data
# ======================================================================================
reload(dataloader)
ds, datasets, dataloaders, combined_eof, scaler_pca = dataloader.load_pcdata(**config)


# %%
# Define model and scheduler
# ======================================================================================
reload(lstm)
# Define model
condition_dim = 12 if config["film"] else None
model = lstm.EncDecLSTM(
    input_dim=combined_eof.n_components,
    hidden_dim=config["hidden_dim"],
    num_layers=config["layers"],
    num_conditions=condition_dim,
    num_tails=config["members"],
    T_max=config["chrono"],
)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.to(device)

# Optimizer and Scheduler
num_iter_per_epoch = len(dataloaders["train"])
num_epochs = config["epochs"]
gradscaler = torch.amp.GradScaler()
optimiser = torch.optim.AdamW(model.parameters(), lr=config["init_lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimiser, T_max=num_epochs * num_iter_per_epoch, eta_min=config["min_lr"]
)

# Loss function
if config["loss_type"] == "crps":
    loss_class = losses.EmpiricalCRPS(reduction="none")
    loss_fn = lambda x_target, x_pred: loss_class(x_target.unsqueeze(dim=1), x_pred)
elif config["loss_type"] == "mse":
    loss_class = nn.MSELoss(reduction="none")
    loss_fn = lambda x_target, x_pred: loss_class(x_target, x_pred.mean(dim=1))
else:
    raise ValueError("Loss type not recognized!")

gamma_scheduler = losses.GammaWeighting(config["gamma"], config["gamma"], 1)

# Per-feature loss weights (SSHA scaled relative to SSTA); all-ones reproduces a plain mean.
feat_weights = losses.variable_loss_weights(list(ds.data_vars), config["n_eof"], {"ssha": config["ssha_weight"]}).to(
    device
)


# Model name
model_type = "CSLSTM" if config["film"] else "LSTM"
lrschedule = (
    f"constlr{config['init_lr']}"
    if config["init_lr"] == config["min_lr"]
    else f"cosinelr{config['init_lr']}-{config['min_lr']}"
)
model_name = (
    f"{model_type}_"
    + "_".join([f"{var}_n{n}" for var, n in zip(ds.data_vars, config["n_eof"])])
    + f"_g{config['gamma']}-{config['loss_type']}_member{config['members']}"
    + f"_nhist_{config['hist']}_nhoriz_{config['train_horiz']}"
    + f"_layers_{config['layers']}_latent{config['hidden_dim']}"
    + f"_{lrschedule}_bs{config['batch_size']}"
    + (f"_wssha{config['ssha_weight']}" if config["ssha_weight"] != 1.0 else "")
    + f"{config['postfix']}"
)
print(model_name, flush=True)

# Check if model exists
model_path = config["path"] + f"/{config['slurm_id']}_" + model_name
if os.path.exists(model_path + "/final_checkpoint.pt"):
    raise ValueError("Model exists! Terminate script.")

# Create directory
if not os.path.exists(model_path):
    print(f"Create directoty {model_path}", flush=True)
    os.makedirs(model_path)
with open(model_path + "/config.json", "w") as f:
    json.dump(config, f)

# Initialize Weights and Biases
runname = model_name
if not config["dry"]:
    wandb.init(config=config, name=model_name, project=config["wandb_project"])

# %%
# Training loop
# ======================================================================================
# Training helper variables
max_hist, max_horiz, train_horiz = config["hist"], config["horiz"], config["train_horiz"]
train_dataloader, val_dataloader = dataloaders["train"], dataloaders["val"]

# Main loop
train_loss, val_loss, val_mse = [], [], []
val_loss_min = 5e5

# Per-lead validation metrics (Niño4 + field-mean indices), computed every N epochs.
val_lag_arr = [1, 3, 6, 9, 12, 15, 18, 21, 24]
val_times = val_dataloader.dataset.data["time"].data
val_monitor = heval.LatentIndexMonitor(combined_eof, scaler_pca, ds, val_times, val_lag_arr, config["members"])
valmetric_every = config["valmetric_every"]

print(f"Training on {device}!", flush=True)
for current_epoch in range(num_epochs):
    tstart = time.time()

    # 1. Validation
    model.eval()
    do_val_metrics = (current_epoch % valmetric_every == 0) or (current_epoch == num_epochs - 1)
    val_frcst, val_time_idx = [], []
    with torch.no_grad():
        vl, mse = 0.0, 0.0
        for sample, aux in val_dataloader:
            hist = 4
            unused = max_hist - hist
            x_input, x_target, _ = sample.to(device).split([hist, max_horiz, unused], dim=1)
            context, _ = aux["month"].to(device=device, dtype=torch.long).split([hist + max_horiz, unused], dim=1)
            # Forward pass
            x_ensemble = model(x_input, context)
            # Loss
            raw_loss = loss_fn(x_target, x_ensemble)
            raw_loss = (raw_loss * feat_weights).sum(dim=2).div(feat_weights.sum()).mean(dim=0)
            gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).to(device).float()
            gamma /= gamma.sum()
            loss = (raw_loss * gamma).sum()
            vl += loss.item()
            mse += (x_ensemble.mean(dim=1) - x_target).pow(2).mean()
            # Collect forecast (normalized PCs) and valid-time indices for index metrics
            if do_val_metrics:
                val_frcst.append(x_ensemble.cpu())
                val_time_idx.append(aux["idx"][:, hist:])

        vl /= len(val_dataloader)
        mse /= len(val_dataloader)
        val_loss.append(vl)
        val_mse.append(mse)

        # Per-lead Niño4 + field-mean metrics over the full validation set (fast operator)
        val_index_log = {}
        if do_val_metrics:
            z_frcst = torch.cat(val_frcst, dim=0)
            time_idx = torch.cat(val_time_idx, dim=0).to(torch.long).numpy()
            val_index_log = val_monitor.compute(z_frcst, time_idx, prefix="val")

    # 2. Training
    model.train()
    tl = 0.0
    unused = max_horiz - train_horiz
    for sample, aux in train_dataloader:
        optimiser.zero_grad()
        # Select random history
        hist = torch.randint(1, max_hist, (1,)).item()
        unused = (max_hist + max_horiz) - (hist + train_horiz)
        x_input, x_target, _ = sample.to(device).split([hist, train_horiz, unused], dim=1)
        context, _ = aux["month"].to(device=device, dtype=torch.long).split([hist + train_horiz, unused], dim=1)
        # Forward pass
        x_ensemble = model(x_input, context)
        # Loss
        raw_loss = loss_fn(x_target, x_ensemble)
        raw_loss = (raw_loss * feat_weights).sum(dim=2).div(feat_weights.sum()).mean(dim=0)
        gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).to(device).float()
        gamma /= gamma.sum()
        loss = (raw_loss * gamma).sum()
        # Backward pass
        gradscaler.scale(loss).backward()
        gradscaler.step(optimiser)
        gradscaler.update()
        tl += loss.item()
        scheduler.step()  # Warning can be ignored, is only caused by gradscaler.step() not being recognized as optimiser.step()

    # Calculate loss for plotting
    tl /= len(train_dataloader)
    train_loss.append(tl)

    # 3. Create checkpoint
    checkpoint = {
        "epoch": current_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mse": val_mse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
    }

    # 4. Print loss
    tend = time.time()
    print(
        f"Epoch {current_epoch}/{num_epochs}, train: {tl:.2e}, "
        + f"val. horiz.: {vl:.2e}, "
        + f"time: {tend - tstart}",
        flush=True,
    )

    # 5. Log it with wandb
    if not config["dry"]:
        wandb.log({"train_loss": tl, "val_loss": vl, "val_mse": mse, "lr": scheduler.get_last_lr()[0], **val_index_log})

    # 6. Save checkpoint
    if model_path is not None:
        if vl < val_loss_min:
            print("Save checkpoint for lowest val. loss!", flush=True)
            torch.save(checkpoint, model_path + "/min_checkpoint.pt")
            val_loss_min = vl
            if val_index_log:
                val_objective = {
                    "epoch": current_epoch,
                    "val_loss": vl,
                    "nino4_acc_mean": val_index_log["val/nino4/acc/mean"],
                    "nino4_acc_per_lag": {
                        int(lag): val_index_log[f"val/nino4/acc/lag{int(lag)}"] for lag in val_lag_arr
                    },
                }
                with open(model_path + "/val_objective.json", "w") as f:
                    json.dump(val_objective, f, indent=2)


# Save model at the end
print("Finished training and save model!", flush=True)
if model_path is not None:
    torch.save(checkpoint, model_path + "/final_checkpoint.pt")


# %%
# Plot loss
# ======================================================================================
fig, ax = plt.subplots()
ax.plot(checkpoint["train_loss"], label=f"training {checkpoint['train_loss'][-1]:.2e}")
ax.plot(checkpoint["val_loss"], label=f"validation horiz. {checkpoint['val_loss'][-1]:.2e}")
ax.set_yscale("log")
ax.set_xlabel("epochs")
ax.set_ylabel("loss")
ax.legend()

plt.savefig(model_path + "/loss.png", dpi=300, bbox_inches="tight")

# %%
# Evaluate model
# ======================================================================================
from eval_lstm import perform_hindcast_evaluation

if config["evaluate"]:
    checkpoint = torch.load(model_path + "/min_checkpoint.pt")
    lag_arr = [1, 3, 6, 9, 12, 15, 18, 21, 24]
    perform_hindcast_evaluation(
        model, checkpoint, ds, dataloaders["test"], scaler_pca, combined_eof, lag_arr, model_path + "/metrics"
    )
