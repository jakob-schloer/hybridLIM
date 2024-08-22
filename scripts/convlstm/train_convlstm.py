"""Training script of convolutional LSTM."""
# %%
import os, time, wandb, argparse
import xarray as xr
import torch
from torch import nn
from importlib import reload
from matplotlib import pyplot as plt
from hyblim import losses
from hyblim.model import convlstm
from hyblim.data import dataloader, eof


PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../../paper.mplstyle")
os.environ["WANDB__SERVICE_WAIT"] = "300"

id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)

# Parameters
# ======================================================================================
def make_args(ipython=False):
    """
     python contextcast_train.py \
        -hist 4 -bs 64 -layers 2 -latent 256 -film -member 16 \
        -horiz 16 -loss crps -gamma 0.65 -epochs 40 -ilr 1e-3 -mlr 1e-6 \ 
        -wandb ContextCast -path ../../output/contextCast/
    """
    if ipython:
        config = dict(
            # Data parameters
            var = ['ssta', 'ssha'],
            num_traindata=None,
            hist = 4,
            batch_size = 64,

            #Model parameters
            latent_dim=256,
            num_layers=2,
            film=True,
            members=16,

            # Training parameters
            epochs=1,      # TODO: Only for testing
            train_horizon=16,    # TODO: Only for testing
            loss_type='normal_crps',
            alpha=1.0,
            gamma = 0.65,
            init_lr = 1e-3,
            min_lr = 1e-6,
            wandb_project='ContextCast',
            dry=True,

            # Saving
            postfix="_test",
            path = PATH + '/../../output/contextCast/'
        )
    else:
        parser = argparse.ArgumentParser()
        # Data
        parser.add_argument('-hist', '--hist', default=4, type=int,
                            help='Length of history.')
        parser.add_argument('-ntrain', '--num_traindata', default=None, type=int,
                            help="Number of training datapoints.")
        parser.add_argument('-bs', '--batch_size', default=64, type=int,
                            help='Batch size.')
        # Model
        parser.add_argument('-layers', '--num_layers', default=2, type=int,
                            help='Number of LSTM layers.')
        parser.add_argument('-latent', '--latent_dim', default=256, type=int,
                            help='Latent dimenison.')
        parser.add_argument("-film","--film", action="store_true",
                            help='If set, conditioning with FiLM on month.')
        parser.add_argument('-members', '--members', default=16, type=int,
                            help='Number of ensemble members.')
        # Training
        parser.add_argument('-horiz', '--train_horizon', default=16, type=int,
                            help='Number of horizon datapoints for training.')
        parser.add_argument('-loss', '--loss_type', default="crps", type=str,
                            help="Loss type: 'mse' or 'crps'.")
        parser.add_argument('-gamma', '--gamma', default=0.65, type=float,
                            help='Weighting of loss, gamma^tau. Defaults to 1.')
        parser.add_argument('-epochs', '--epochs', default=30, type=int,
                            help='Number of epochs.')
        parser.add_argument('-ilr', '--init_lr', default=1e-3, type=float,
                            help='Initial learning rate.')
        parser.add_argument('-mlr', '--min_lr', default=1e-6, type=float,
                            help='Minimum learning rate for Cosineannealing.')
        parser.add_argument('-wandb', '--wandb_project', default="ContextCast",
                            type=str, help='Wandb project name.')
        parser.add_argument("-dry","--dry", action="store_true",
                            help='If set, dry run.')
        # Save model
        parser.add_argument('-path', '--path', default=PATH + '/../../output/contextCast/',
                            type=str, help='Modelpath.')
        parser.add_argument('-postfix', '--postfix', default="", type=str,
                            help="Postfix to model folder, e.g. '_id_1'.")

        config = vars(parser.parse_args())

    config['horiz'] = 24
    config['slurm_id'] = os.environ.get('SLURM_JOB_ID', 0)

    return config 

# Run in ipython mode
config = make_args(ipython=False)

# %%
# Load data
# ======================================================================================
data_path = PATH + '/../../data/processed_datasets/CESM2_piControl/'
 

filepaths = {
        'ssta': data_path +"/ssta_lat-31_32_lon130_-70_gr1.0_norm-zscore.nc",
        'ssha': data_path +"/ssha_lat-31_32_lon130_-70_gr1.0_norm-zscore.nc",
}
f_lsm = data_path + '/land-sea-mask_lat-31_32_lon130_-70_gr1.0.nc'

ds, datasets, dataloaders = dataloader.load_stdata(filepaths, **config)
train_dataloader, val_dataloader = dataloaders['train'], dataloaders['val']

#Get lsm for loss masking
land_area_mask = xr.open_dataset(f_lsm)['sftlf']
lsm = torch.logical_not(torch.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data)).to(device)

# %%
# Defining model and training parameters 
# ======================================================================================
reload(losses)
# Training parameters
num_iter_per_epoch = len(train_dataloader)
num_epochs = config['epochs']
gamma, train_horizon = config['gamma'], config['train_horizon']
init_lr = config['init_lr'] * config['batch_size']/64
min_lr = config['min_lr'] * config['batch_size']/64

# Model
data_dim = len(list(ds.data_vars))
condition_num = 12 if config['film'] else -1
model = models.ContextCast(
    input_dim=data_dim, encoder_dim=config['latent_dim'], 
    decoder_dim=config['latent_dim'], output_dim=data_dim,
    num_layers=config['num_layers'], num_conditions=condition_num,
    num_tails=config['members'], k_conv=7, patch_size=(1, 8, 8)
)

print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
model.to(device)

# Optimizer and Scheduler
gradscaler = torch.cuda.amp.GradScaler()
optimiser = torch.optim.AdamW(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimiser, T_max=num_epochs*num_iter_per_epoch, eta_min=min_lr
)

# Loss function
if config['loss_type']=='crps':
    loss_class = losses.EmpiricalCRPS(reduction='none', dim=1)
    loss_fn = lambda x, x_hat: loss_class(x.unsqueeze(1), x_hat)
elif config['loss_type']=='normal_crps':
    loss_class = losses.NormalCRPS(reduction='none', dim=1, mode='ensemble')
    loss_fn = lambda x, x_hat: loss_class(x, x_hat)
elif config['loss_type']=='mse':
    loss_fn = nn.MSELoss(reduction='none')
elif config['loss_type']=='crps_mse':
    crps_class = losses.EmpiricalCRPS(reduction='none')
    mse_class = nn.MSELoss(reduction='none')
    alpha = config['alpha']
    loss_fn = lambda x, x_hat: (crps_class(x.unsqueeze(1), x_hat) 
                                + alpha * mse_class(x.unsqueeze(1), x_hat).mean(dim=1)) 
else:
    raise ValueError('Loss type not recognized!')

gamma_scheduler = losses.GammaWeighting(gamma, gamma, 1, device)

# Model name
model_type = "CSContextCast" if config['film'] else "ContextCast"
lrschedule = f"constlr{config['init_lr']}" if config['init_lr']==config['min_lr'] else  f"cosinelr{config['init_lr']}-{config['min_lr']}"
model_name = (f"{model_type}_g{gamma}-{config['loss_type']}_hist{config['hist']}_horiz{train_horizon}" 
              + f"_layer{config['num_layers']}_latent{config['latent_dim']}_mem{config['members']}"
              + f"_{lrschedule}_bs{config['batch_size']}"
              + f"{config['postfix']}")
print(model_name, flush=True)

# Check if model exists
model_path = os.path.join(config['path'], f"{config['slurm_id']}_" + model_name)
if os.path.exists(os.path.join(model_path, 'final_checkpoint.pt')):
    raise ValueError('Model exists! Terminate script.')

# Create directory
if not os.path.exists(model_path):
    print(f"Create directoty {model_path}", flush=True)
    os.makedirs(model_path)
torch.save(dict(config), model_path + "/config.pt")

# Weights and Biases
if not config['dry']:
    config['path'] = model_path + model_name
    wandb.init(config=config, name=model_name, project=config['wandb_project'])

# %%
# Main training loop 
# ======================================================================================
print(f'Training new model {model_name}', flush=True)

#Training helper variables
train_loss_tracker, val_loss_tracker, val_mse_tracker = [], [], []
cycle_idx = 0
horizon = config['horiz']
history = config['hist']
unused = horizon - train_horizon

#Main loop
for current_epoch in range(num_epochs):

    start_time_epoch = time.time()
    #Validation loop
    model.eval()
    with torch.no_grad():
        vl = 0
        mse = 0
        for sample, aux in val_dataloader:
            x, y = sample.to(device).split([history, horizon], dim = 2)
            c = aux['month'].to(device = device, dtype = torch.long).argmax(dim = -1)
            
            x_pred = model(x, context=c)
            # Loss
            raw_loss = loss_fn(y, x_pred)[:, :, :, lsm]
            raw_loss = raw_loss.mean(dim=[0, 1, 3])
            gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).float()
            gamma /= gamma.sum()
            loss = (raw_loss * gamma).sum()
            vl += loss.item()
            x_mu = x_pred.mean(dim=1)
            mse += (x_mu - y).pow(2)[:, :, :, lsm].mean()

        vl /= len(val_dataloader)
        mse /= len(val_dataloader)
        val_loss_tracker.append(vl)
        val_mse_tracker.append(mse)
    

    #Train loop
    model.train()
    tl = 0
    for sample, aux in train_dataloader:
        x, y, _ = sample.to(device).split([history, train_horizon, unused], dim = 2)
        c, _ = aux['month'].to(device = device, dtype = torch.long).argmax(
            dim = -1).split([history+train_horizon, unused], dim=-1) #Convert from one-hot to month index
        optimiser.zero_grad()

        with torch.cuda.amp.autocast(): #Mixed precision training for faster training, maybe not needed
            x_pred = model(x, context=c)
            # Loss
            raw_loss = loss_fn(y, x_pred)[:, :, :, lsm]
            raw_loss = raw_loss.mean(dim=[0, 1, 3])
            gamma = gamma_scheduler(raw_loss.shape[0], current_epoch).float()
            gamma /= gamma.sum()
            loss = (raw_loss * gamma).sum()

        gradscaler.scale(loss).backward()
        gradscaler.step(optimiser)
        gradscaler.update()
        tl += loss.item()
        scheduler.step() #Warning can be ignored, is only caused by gradscaler.step() not being recognized as optimiser.step()

    #Calculate loss for plotting
    tl /= len(train_dataloader)
    train_loss_tracker.append(tl)

    #Save checkpoint
    checkpoint = {
                'train_loss': train_loss_tracker,
                'val_loss': val_loss_tracker,
                'val_mse': val_mse_tracker,
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }

    # Log it with wandb
    if not config['dry']:
        wandb.log(
            {"train_loss": tl,
             "val_loss": vl, 
             "val_mse": mse,
             'lr': scheduler.get_last_lr()[0],
             'max_memory': torch.cuda.max_memory_allocated(device)/1e9,
             }
        )
    #Print progress
    print(f'Epoch: {current_epoch}, Train Loss: {tl:,.2e}, Val Loss: {vl:,.2e}', flush=True)
    print(f'Elapsed time:  {(time.time() - start_time_epoch)/60 :.2f} minutes, max. mem. {torch.cuda.max_memory_allocated(device)/1e6}', flush=True)

    # Save checkpoint
    if (current_epoch + 1) % 5 == 0:
        torch.save(checkpoint, model_path + f'/checkpoint_{current_epoch+1}.pt')
    

print("Finished training!", flush=True)
torch.save(checkpoint, model_path + '/final_checkpoint.pt')

# %%
# Hindcast
# ======================================================================================
reload(hlp)
print("Hindcast validation set!", flush=True)
history, horizon = config['hist'], config['horiz']
x_hindcast, time_idx = hlp.hindcast(
    val_dataloader, model, device, history, horizon
)

# Compute metric
# ======================================================================================
val_ds = datasets['val'].dataset
lag_arr = [1, 3, 6, 9, 12, 15, 18, 24]

verification_per_gridpoint, verification_per_time, nino_indices = hlp.evaluation_metric(
    x_hindcast, time_idx, val_ds, land_area_mask, lag_arr
)

# Save metrics to file
# ======================================================================================
print("Save evaluation to file!", flush=True)
torch.save(verification_per_gridpoint, model_path + "/metrics_grid.pt")
torch.save(verification_per_time, model_path + "/metrics_time.pt")
torch.save(nino_indices, model_path + "/nino_indices.pt")
# %%
