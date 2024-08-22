'''LIM + LSTM in pca space. 

@Author  :   Jakob Schlör 
@Time    :   2022/11/22 16:55:52
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import time, os, argparse, wandb, json
import xarray as xr
import torch
from torch import nn
import matplotlib.pyplot as plt
from importlib import reload
from hyblim import losses
from hyblim.model import lstm
from hyblim.data import dataloader, eof

PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["WANDB__SERVICE_WAIT"] = "300"
id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)

# configeters
# ======================================================================================
def make_args(ipython=False):
    """ Run script:
    
    python lim_declstm_train.py -v ts zos -eof 20 -horiz 24 -lim cslim -cond -path ./ -ep 50 -lr 0.0001 -postfix _id_1

    """
    if ipython:
        config = dict(
            # Data
            vars=['ssta', 'ssha'],
            num_traindata=None,
            batch_size=8,
            n_eof=[20,10],
            # Model
            layers=2,
            hidden_dim=32,
            film = True,
            # Training
            train_horiz=16,
            loss_type='crps',
            gamma=0.65, #0.9,
            epochs = 5,
            init_lr = 5e-3,
            min_lr = 5e-6,
            wandb_project='LIM+LSTM',
            dry=True,
            sweep=False,
            # Saving
            path=PATH + f"/../../models/limlstm/",
            postfix="",
        )
    else:
        parser = argparse.ArgumentParser()
        # Data
        parser.add_argument('-v', '--vars', nargs='+', default=['ssta', 'ssha'],
                            help='Variable names used.')
        parser.add_argument('-eof', '--n_eof', nargs='+', default=[20, 10],
                            help="Number of eof components for LSTM, the LIM uses 20 only.")
        parser.add_argument('-ntrain', '--num_traindata', default=None, type=int,
                            help="Number of training datapoints.")
        parser.add_argument('-horiz', '--train_horiz', default=16, type=int,
                            help='Number of horizon datapoints.')
        parser.add_argument('-batch', '--batch_size', default=96, type=int,
                            help='Batch size.')
        # Model
        parser.add_argument('-l', '--layers', default=1, type=int,
                            help='Number of LSTM layers.')
        parser.add_argument('-hidden', '--hidden_dim', default=64, type=int,
                            help='Dimension of hidden dimension.')
        parser.add_argument("-film", "--film", action="store_true",
                            help='If set, conditioning with FiLM on month.')
        # Training
        parser.add_argument('-loss', '--loss_type', default="crps", type=str,
                            help="Name of loss used for training, i.e. 'weighted_mse', 'mse' ")
        parser.add_argument('-gamma', '--gamma', default=0.65, type=float,
                            help='Weighting of loss, gamma^tau.')
        parser.add_argument('-epochs', '--epochs', default=30, type=int,
                            help='Number of epochs.')
        parser.add_argument('-ilr', '--init_lr', default=5e-3, type=float,
                            help='Initial learning rate.')
        parser.add_argument('-mlr', '--min_lr', default=5e-6, type=float,
                            help='Minimum learning rate for Cosineannealing.')
        parser.add_argument('-wandb', '--wandb_project', default="LIM+LSTM",
                            type=str, help='Wandb project name.')
        parser.add_argument("-dry", "--dry", action="store_true",
                            help='If set, dry run.')
        # Save model
        parser.add_argument('-path', '--path', default=PATH + f"/../../output/lim+lstm/",
                            type=str, help='Modelpath.')
        parser.add_argument('-postfix', '--postfix', default="", type=str,
                            help="Postfix to model folder, e.g. '_id_1'.")
        parser.add_argument('-eval', '--evaluate', action='store_true', help='Evaluate model.')
        config = vars(parser.parse_args())

    path_to_data = {
        'ssta': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        'ssha': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
    }
    config['datapaths'] = {}
    for var in config['vars']:
        config['datapaths'][var] = path_to_data[var]
    config['lsm_path'] = PATH + "/../../data/land_sea_mask_common.nc"

    if config['num_traindata'] is not None:
        config['lim_path'] = PATH + f"/../../models/lim/cslim_{'-'.join(map(str, config['vars']))}/num_traindata/n_{config['num_traindata']}/cslim_hindcast_{'-'.join(map(str, config['vars']))}_eof20"
    else:
        config['lim_path'] = PATH + f"/../../models/lim/cslim_{'-'.join(map(str, config['vars']))}/cslim_hindcast_{'-'.join(map(str, config['vars']))}_eof20"

    config['horiz'] = 24
    config['slurm_id'] = os.environ.get('SLURM_JOB_ID', 0000000)
    config['chrono'] = 5
    config['name'] = 'LIM+LSTM'

    assert len(config['n_eof']) == len(config['vars']) 

    return config

# Get configs
config = make_args(ipython=True)

# %%
# Create training and validation dataset
# ======================================================================================
reload(dataloader)
lim_hindcast = xr.concat([
   xr.open_dataset(config['lim_path'] + "_train.nc"),
   xr.open_dataset(config['lim_path'] + "_val.nc"),
   xr.open_dataset(config['lim_path'] + "_test.nc"),
], dim='time')['z'].sel(lag=slice(1, None))

# Create dataset
ds, datasets, dataloaders, combined_eofa, normalizer_pca = dataloader.load_pcdata_lim_ensemble(
    lim_hindcast, **config
) 
lim_input, target, label = datasets['train'][0]
members, n_horiz, n_feat =lim_input.shape

# %%
# Define model and training parameters
# ======================================================================================
reload(lstm)
num_condition = 12 if config['film'] else -1
model = lstm.ResidualLSTM(
    input_dim=combined_eofa.n_components, hidden_dim=config['hidden_dim'],
    num_conditions=num_condition, num_layers=config['layers'],
    T_max=config['chrono']
)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
model.to(device)

# Optimizer and Scheduler
num_epochs, num_iter_per_epoch = config['epochs'], len(dataloaders['train']) 
gradscaler = torch.amp.GradScaler()
optimiser = torch.optim.AdamW(model.parameters(), lr=config['init_lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimiser, T_max=num_epochs*num_iter_per_epoch, eta_min=config['min_lr']
)

# Loss function
if config['loss_type']=='crps':
    loss_class = losses.EmpiricalCRPS(reduction='none')
    loss_fn = lambda x_target, x_pred: loss_class(x_target.unsqueeze(dim=1), x_pred)
elif config['loss_type']=='mse':
    loss_class = nn.MSELoss(reduction='none')
    loss_fn = lambda x_target, x_pred: loss_class(x_target, x_pred.mean(dim=1))
else:
    raise ValueError('Loss type not recognized!')

gamma_scheduler = losses.GammaWeighting(config['gamma'], config['gamma'], 1, device)

# Model name
model_type = "LIM-LSTM" 
lrschedule = f"constlr{config['init_lr']}" if config['init_lr']==config['min_lr'] else  f"cosinelr{config['init_lr']}-{config['min_lr']}"
model_name = (f"{model_type}_"
              + "_".join([f"{var}_n{n}" for var, n in zip(ds.data_vars, config['n_eof'])])
              + f"_g{config['gamma']}-{config['loss_type']}_member{members}"
              + f"_nhoriz_{config['train_horiz']}"
              + f"_layers_{config['layers']}_latent{config['hidden_dim']}"
              + f"_{lrschedule}_bs{config['batch_size']}"
              + f"{config['postfix']}")
print(model_name, flush=True)

# Check if model exists
model_path = config['path'] + f"/{config['slurm_id']}_" + model_name
if os.path.exists(model_path + '/final_checkpoint.pth'):
    raise ValueError('Model exists! Terminate script.')

# Create directory
if not os.path.exists(model_path):
    print(f"Create directoty {model_path}", flush=True)
    os.makedirs(model_path)
torch.save(dict(config), model_path + "/config.pt")


# Log to wandb
# Initialize Weights and Biases
runname = model_name 
if not config['dry'] and not config['sweep']:
    wandb.init(config=config, name=model_name, project=config['wandb_project'])
# %%
# Main training loop
# ======================================================================================
train_dataloader, val_dataloader = dataloaders['train'], dataloaders['val']
train_horizon = config['train_horiz']
num_epochs = config['epochs']
unused = len(lim_hindcast['lag']) - train_horizon

# Loss trackers
train_loss, val_loss, val_mse = [], [], []
val_loss_min = 1e5

print(f"Training on {device}!", flush=True)
for current_epoch in range(num_epochs):
    tstart = time.time()

    # 1. Validation
    model.eval()
    with torch.no_grad():
        vl, mse = 0, 0
        for lim_input, target, l in val_dataloader:
            x_input, x_target = lim_input.to(device), target.to(device) 
            context = l['month'].to(device, dtype=torch.long) if config['film'] else None
            # Prediction
            x_hat = model(x_input, context)
            # Loss
            raw_loss = loss_fn(x_target, x_hat)
            raw_loss = raw_loss.mean(dim=[0, 2])
            gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).float()
            gamma /= gamma.sum()
            loss = (raw_loss * gamma).sum()
            vl += loss.item()
            x_mu = x_hat.mean(dim=1)
            mse += (x_mu - x_target).pow(2).mean().item()

        vl /= len(val_dataloader)
        mse /= len(val_dataloader)
        val_loss.append(vl)
        val_mse.append(mse)

    # 2. Training
    model.train()
    tl = 0
    for lim_input, target, l in train_dataloader:
        optimiser.zero_grad()
        # Split and to_device
        x_input, _ = lim_input.to(device).split([train_horizon, unused], dim=2) 
        x_target, _ = target.to(device).split([train_horizon, unused], dim=1) 
        context, _ = l['month'].to(device, dtype=torch.long).split([train_horizon, unused], dim=-1) if config['film'] else (None, None)
        # Prediction
        x_hat = model(x_input, context)
        # Loss
        raw_loss = loss_fn(x_target, x_hat)
        raw_loss = raw_loss.mean(dim=[0, 2])
        gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).float()
        gamma /= gamma.sum()
        loss = (raw_loss * gamma).sum()
        # Backpropagation
        gradscaler.scale(loss).backward()
        gradscaler.step(optimiser)
        gradscaler.update()
        tl += loss.item()
        # Scheduler
        # Warning can be ignored, is only caused by gradscaler.step() not being recognized as optimiser.step()
        if scheduler is not None:
            scheduler.step()

    tl /= len(train_dataloader)
    train_loss.append(tl)

    # 3. Create checkpoint
    checkpoint = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_mse': val_mse,
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }

    # 4. Print loss
    tend = time.time()
    print(f"Epoch {current_epoch}, train: {tl:.2e}, "
          + f"val.: {vl:.2e}, "
          + f"time: {tend - tstart}", flush=True)
    
    # 5. Log it with wandb
    if not config['dry']:
        wandb.log(
            {"train_loss": tl, "val_loss": vl, 'val_mse': mse,
             'lr': scheduler.get_last_lr()[0]}
        )

    # 6. Save checkpoint
    if model_path is not None:
        if (vl < val_loss_min):
            print("Save checkpoint for lowest val. loss!", flush=True)
            torch.save(checkpoint, model_path + f"/min_checkpoint.pt")
            val_loss_min = vl 

# Save model at the end
print("Finished training and save model!", flush=True)
if model_path is not None:
    torch.save(checkpoint, model_path + f"/final_checkpoint.pt")


# %%
# Plot loss
# ======================================================================================
fig, ax = plt.subplots()
ax.plot(checkpoint['train_loss'],
        label=f"training {checkpoint['train_loss'][-1]:.2e}")
ax.plot(checkpoint['val_loss'],
        label=f"validation horiz. {checkpoint['val_loss'][-1]:.2e}")
ax.set_yscale('log')
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend()

plt.savefig(model_path + f"/loss.png", dpi=300, bbox_inches='tight')

