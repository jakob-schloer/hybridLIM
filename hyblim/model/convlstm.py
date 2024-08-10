'''LSTM models with 2D inputs. 

@Author  :   Jakob Schlör 
@Time    :   2022/09/11 14:47:32
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch as th
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from hyblim.model import modules


def count_parameters(model, min_param_size = 512):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_params = parameter.numel()
        total_params += param_params
        if param_params > min_param_size:
            print(f"{name}: {param_params}")
    print(f"Total Parameters: {total_params}")


class SwinLSTM(nn.Module):
    def __init__(self, x_channels: int, h_channels: int, kernel_size: int = 7, activation_fn = nn.Tanh()):
        '''
        :param x_channels: Input channels   
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        conv_channels = x_channels + h_channels
        self.spatial_mixing = nn.Conv2d(conv_channels, conv_channels, kernel_size, padding='same', groups= conv_channels)
        self.channel_mixing = nn.Conv2d(conv_channels, 4 * h_channels, 1)

        self.separate_gates = Rearrange('b (gates c) h w -> gates b c h w', gates = 4, c = h_channels)
        
        self.activation_fn = activation_fn
        self.gate_norm = nn.GroupNorm(num_channels = conv_channels, num_groups = 4, affine = False)
        self.output_norm = nn.GroupNorm(num_channels = h_channels, num_groups = 1, affine = True)

        nn.init.dirac_(self.spatial_mixing.weight)
        nn.init.dirac_(self.channel_mixing.weight)
        nn.init.ones_(self.channel_mixing.bias[:conv_channels]) * 2

    def forward(self, x, h, c, conditioning = None):
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        z = th.cat((x, h), dim = 1) if x is not None else h # TODO: Change to x + h
        z = self.spatial_mixing(z)
        z = self.gate_norm(z)
        z = self.channel_mixing(z)
        if conditioning is not None:
            a, b = einops.rearrange(conditioning, 'b (split c) -> split b c () ()', split = 2)
            z = z * (1 + a) + b

        i, f, o, g = self.separate_gates(z) 
        c = th.sigmoid(f) * c + th.sigmoid(i) * th.tanh(g)
        h = th.sigmoid(o) * self.activation_fn(self.output_norm(c))
        return h, c
    
class SwinLSTMNet(nn.Module):
    def __init__(self,
                 data_dim: int,
                 latent_dim: int,
                 patch_size: tuple = (4,4),
                 num_layers: int = 2,
                 residual_forecast: bool = False,
                 conditioning_num: int = -1,
                 k_conv: int = 7,
                 num_members: int = 16,
                 ) -> None:
        '''
        ElNet: The SwinLSTMNet model.
        Args:
            data_dim: The number of input channels
            latent_dim: The dimension of the latent space.
            patch_size: The size of the latent patches.
            num_layers: The number of LSTM layers in the model.
            residual_forecast: Whether to use residual connections in the forecast.
            k_conv: The kernel size of the convolutional layers.
            expansion_factor: The expansion factor of the processing blocks.
            activation_fn: The activation function of the processing blocks.'''
        
        super().__init__()
        #define model attributes
        self.num_layers = num_layers
        self.use_residual = residual_forecast #Have not tested this option carefully yet
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.use_film = conditioning_num > 0
        self.num_members = num_members
        #define model layers
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Sequential(
            nn.Conv3d(data_dim, latent_dim, kernel_size=(1,7,7), stride=(1,2,2),
                      padding=(0,3,3)), 
	        nn.ReLU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=(1,3,3), stride = (1,2,2),
                      padding=(0,1,1))
        )
        for i in range(num_layers):
            self.processor[f'encoder_lstm_{i}'] = SwinLSTM(latent_dim, latent_dim, k_conv)
            if i == 0:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(0, latent_dim, k_conv)
            else:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(latent_dim, latent_dim, k_conv)
        
        self.processor['to_data'] = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, data_dim, kernel_size=(1,3,3), 
                               stride = (1,2,2), padding=(0,1,1), output_padding=(0,1,1)),
            nn.ReLU(),
            modules.MultiHeadConvTranspose3d(
                data_dim, data_dim, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                output_padding=(0,1,1), num_heads=self.num_members 
            )
        )
                
        if self.use_film:
            self.processor['embedding'] = nn.Embedding(
                num_embeddings = conditioning_num, embedding_dim = 8 * latent_dim) 
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)

    def forward(self, x, horizon = 1, conditioning = None):
        '''
        Forward pass of the model.
        Args:
            x: Input tensor.
            horizon: The number of steps to predict.
            conditioning: Auxiliary conditioning tensor.'''
        batch, _, history, height, width = x.shape
        h = [th.zeros((batch, self.latent_dim, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        c = [th.zeros((batch, self.latent_dim, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        #encoder
        patches = self.processor['to_latent'](x)
        for t in range(history):
            z = patches[:, :, t]
            u = self.processor['embedding'](conditioning)[:,t] if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'encoder_lstm_{i}'](z, h[i], c[i], conditioning=u) 
                z += h[i]
        #decoder
        z_out = []
        for t in range(horizon):
            z = None
            u = self.processor['embedding'](conditioning)[:,history+t] if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'decoder_lstm_{i}'](z, h[i], c[i], conditioning=u)    
                z = h[i] if z is None else z + h[i]
            z_out.append(z)
        z_out = th.stack(z_out, dim = 2)

        # To data space
        x_pred = self.processor['to_data'](z_out)

        return x_pred


class ResidualSwinLSTMNet(nn.Module):
    def __init__(self,
                 data_dim: int,
                 latent_dim: int,
                 patch_size: tuple = (4,4),
                 num_layers: int = 2,
                 conditioning_num: int = -1,
                 k_conv: int = 7,
                 ) -> None:
        '''
        ElNet: The ElNet model.
        Args:
            data_dim: The number of input channels
            latent_dim: The dimension of the latent space.
            patch_size: The size of the latent patches.
            num_layers: The number of LSTM layers in the model.
            k_conv: The kernel size of the convolutional layers.
            expansion_factor: The expansion factor of the processing blocks.
            activation_fn: The activation function of the processing blocks.'''
        
        super().__init__()
        #define model attributes
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.use_film = conditioning_num > 0
        #define model layers
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Sequential(
            nn.Conv2d(data_dim, latent_dim, patch_size, stride = patch_size),
            nn.GroupNorm(num_channels = latent_dim, num_groups = 1, affine=True),
        )
        for i in range(num_layers):
            self.processor[f'decoder_lstm_{i}'] = SwinLSTM(latent_dim, latent_dim, k_conv)
                
        self.processor['to_data'] = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size= patch_size, stride = patch_size),
            nn.GroupNorm(num_channels = latent_dim, num_groups = 1, affine=True),
            nn.Conv2d(latent_dim, data_dim, 1)
        )

        if self.use_film:
            self.processor['embedding'] = nn.Embedding(num_embeddings = conditioning_num, embedding_dim = 8 * latent_dim) #Maybe *2 and share per gate?
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)

    def forward(self, x, conditioning = None):
        '''
        Forward pass of the model.
        Args:
            x: Input tensor.
            horizon: The number of steps to predict.
            conditioning: Auxiliary conditioning tensor.'''
        batch, _, horizon, height, width = x.shape
        h = [th.zeros((batch, self.latent_dim, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        c = [th.zeros((batch, self.latent_dim, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        #decoder
        out = []
        for t in range(horizon):
            x_input = x[:, :, t]
            z = self.processor['to_latent'](x_input)
            u = self.processor['embedding'](conditioning)[:,t] if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'decoder_lstm_{i}'](z, h[i], c[i], conditioning = u) 
                z = h[i]
            x_hat = x_input + self.processor['to_data'](z) 
            out.append(x_hat)

        #stack time dimension
        out = th.stack(out, dim = 2)
        return out