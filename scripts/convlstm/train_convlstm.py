"""Training script of convolutional LSTM."""
# %%
import os, time, wandb, argparse, json
import xarray as xr
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from importlib import reload
from matplotlib import pyplot as plt
from hyblim import losses
from hyblim.model import convlstm
from hyblim.data import dataloader, eof

PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["WANDB__SERVICE_WAIT"] = "300"

id = int(os.environ.get('SLURM_LOCALID', 0))

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
            vars = ['ssta', 'ssha'],
            num_traindata=1200,
            hist = 12,
            batch_size = 16,

            #Model parameters
            num_channels=256,
            num_layers=2,
            film=True,
            members=16,

            # Training parameters
            epochs=3,      # TODO: Only for testing
            train_horiz=16,    # TODO: Only for testing
            loss_type='crps',
            gamma = 0.65,
            init_lr = 5e-3,
            min_lr = 1e-6,
            wandb_project='SwinLSTM',

            # Saving
            postfix="_test",
            path = PATH + '/../../models/convlstm/'
        )
    else:
        parser = argparse.ArgumentParser()
        # Data
        parser.add_argument('-vars', '--vars', nargs='+', default=['ssta', 'ssha'],
                            help='Variables to use.')
        parser.add_argument('-hist', '--hist', default=12, type=int,
                            help='Length of history.')
        parser.add_argument('-ntrain', '--num_traindata', default=None, type=int,
                            help="Number of training datapoints.")
        parser.add_argument('-batch', '--batch_size', default=64, type=int,
                            help='Batch size.')
        # Model
        parser.add_argument('-layers', '--num_layers', default=2, type=int,
                            help='Number of LSTM layers.')
        parser.add_argument('-channels', '--num_channels', default=256, type=int,
                            help='Latent dimenison.')
        parser.add_argument("-film","--film", action="store_true",
                            help='If set, conditioning with FiLM on month.')
        parser.add_argument('-members', '--members', default=16, type=int,
                            help='Number of ensemble members.')
        # Training
        parser.add_argument('-horiz', '--train_horiz', default=16, type=int,
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

    path_to_data = {
        'ssta': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        'ssha': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
    }
    config['datapaths'] = {}
    for var in config['vars']:
        config['datapaths'][var] = path_to_data[var]

    config['lsm_path'] = PATH + "/../../data/land_sea_mask_common.nc"
    config['horiz'] = 24
    config['name'] = 'SwinLSTM'
    config['slurm_id'] = os.environ.get('SLURM_JOB_ID', 0000000)

    return config 

class ConvLSTMTrainer(pl.LightningModule):
    def __init__(self, config):
        super(ConvLSTMTrainer, self).__init__()
        # Save configuration for later access
        self.save_hyperparameters()
        self.config = config

        # Define the model
        data_channels = config['input_dim']
        condition_num = 12 if config['film'] else -1
        self.model = convlstm.EncDecSwinLSTM(
                 input_dim=data_channels,
                 num_channels=config['num_channels'],
                 output_dim=data_channels,
                 patch_size=(4,4),
                 num_layers= config['num_layers'],
                 num_conditions=condition_num,
                 num_tails=config['members']
        ) 


        # Loss function
        if self.config['loss_type'] == 'crps':
            self.loss_class = losses.EmpiricalCRPS(reduction='none', dim=1)
            self.loss_fn = lambda x, x_hat: self.loss_class(x.unsqueeze(1), x_hat)
        elif self.config['loss_type'] == 'normal_crps':
            self.loss_class = losses.NormalCRPS(reduction='none', dim=1, mode='ensemble')
            self.loss_fn = lambda x, x_hat: self.loss_class(x, x_hat)
        elif self.config['loss_type'] == 'mse':
            self.loss_fn = torch.nn.MSELoss(reduction='none')
        elif self.config['loss_type'] == 'crps_mse':
            crps_class = losses.EmpiricalCRPS(reduction='none')
            mse_class = torch.nn.MSELoss(reduction='none')
            alpha = self.config['alpha']
            self.loss_fn = lambda x, x_hat: (crps_class(x.unsqueeze(1), x_hat) 
                                             + alpha * mse_class(x.unsqueeze(1), x_hat).mean(dim=1))
        else:
            raise ValueError('Loss type not recognized!')

        # Decaying weight for loss over lead time
        self.gamma_scheduler = losses.GammaWeighting(self.config['gamma'], self.config['gamma'], 1)
        #Get lsm for loss masking
        land_area_mask = xr.open_dataset(config['lsm_path'])['lsm']
        self.lsm = torch.logical_not(torch.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data)).to(self.device)
        
    def forward(self, x, context):
        return self.model(x, context=context)

    
    def _step(self, batch, batch_idx, history, horizon):
        sample, aux = batch
        n_batch, n_vars, n_time, n_lat, n_lon = sample.shape 

        unused = n_time - history - horizon
        
        x, y, _ = sample.split([history, horizon, unused], dim=2)
        context, _ = aux['month'].to(dtype=torch.long).split(
            [history + horizon, unused], dim=-1
        )

        x_pred = self.model(x, context=context)
        raw_loss = self.loss_fn(y, x_pred)[:, :, :, self.lsm]
        raw_loss = raw_loss.mean(dim=[0, 1, 3])
        gamma = self.gamma_scheduler(raw_loss.shape[0], self.current_epoch).to(self.device).float()
        gamma /= gamma.sum()
        loss = (raw_loss * gamma).sum()

        x_mu = x_pred.mean(dim=1)
        mse = (x_mu - y).pow(2)[:, :, :, self.lsm].mean()

        return x_pred, loss, mse

    def training_step(self, batch, batch_idx):
        hist = torch.randint(1, self.config['hist'], (1,)).item()
        x_pred, loss, _ = self._step(batch, batch_idx, hist, self.config['train_horiz'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        hist = 4 # TODO: This should not be hardcoded
        x_pred, loss, mse = self._step(batch, batch_idx, hist, self.config['horiz'])

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_mse', mse, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Scale learning rate with batch size and number of GPUs
        base_batch_size = 64 
        effective_batch_size = self.config['batch_size'] * self.trainer.world_size
        base_lr = self.config['init_lr'] 
        scaled_lr = base_lr * (effective_batch_size / base_batch_size)


        optimizer = torch.optim.AdamW(self.model.parameters(), lr=scaled_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.config['epochs'], eta_min=self.config['min_lr']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
         }



# %%
# ======================================================================================
# Run in ipython mode
ipython = False
config = make_args(ipython)

# %%
# Load data
ds, datasets, dataloaders = dataloader.load_stdata(**config)
config['input_dim'] = len(list(ds.data_vars))

# %%
# Instantiate the Lightning Module
model = ConvLSTMTrainer(config=config)

# Model name
lrschedule = f"constlr{config['init_lr']}" if config['init_lr']==config['min_lr'] else  f"cosinelr{config['init_lr']}-{config['min_lr']}"
model_name = (f"ConvLSTM_"
              + "_".join([var for var in list(ds.data_vars)])
              + f"_g{config['gamma']}-{config['loss_type']}_member{config['members']}"
              + f"_nhist_{config['hist']}_nhoriz_{config['train_horiz']}"
              + f"_layers_{config['num_layers']}_ch{config['num_channels']}"
              + f"_{lrschedule}_bs{config['batch_size']}"
              + f"{config['postfix']}")
model_path = config['path'] + f"/{config['slurm_id']}_" +  model_name

# Create directory
if not os.path.exists(model_path):
    print(f"Create directoty {model_path}", flush=True)
    os.makedirs(model_path)
with open(model_path + "/config.json", 'w') as f:
    json.dump(config, f)

# Initialize Weights and Biases
wandb_logger = WandbLogger(
    project=config['wandb_project'],
    name=model_name,
    log_model=True
)

# Callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', 
    dirpath=model_path,
    filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1, 
    save_last=True,
    mode='min',
    save_weights_only=False,
)

# Trainer
trainer = pl.Trainer(
    max_epochs=config['epochs'],
    accelerator='gpu',
    devices=torch.cuda.device_count(), 
    precision=16,  # Mixed precision
    strategy="ddp_notebook" if ipython else "ddp",
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)

# Start training
trainer.fit(model, dataloaders['train'], dataloaders['val'])

# %%
