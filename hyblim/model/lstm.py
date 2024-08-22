'''LSTM models with 2D inputs. 

@Author  :   Jakob Schlör 
@Time    :   2022/09/11 14:47:32
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    '''LSTMCell for 1d inputs with FiLM layer for conditioning.
    
    Args:
        input_dim (int): Input dimensions 
        hidden_dim (int): Hidden dimensions
        condition_dim (int): Condition dimensions
    '''

    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 T_max: int = -1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 4),
            nn.GroupNorm(num_channels=4*hidden_dim, num_groups=4, affine=False)
        )
        self.output_norm = nn.GroupNorm(num_channels=hidden_dim, num_groups=1, affine=True)

        if isinstance(T_max, int) and T_max > 1:
            self._chrono_init_(T_max)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, context: torch.Tensor=None) -> torch.Tensor:
        '''LSTM forward pass.
        Args:
            x (torch.Tensor): Input
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
            context (torch.Tensor): Conditioning. Default: None 
        Returns:
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
        '''
        z = torch.cat((x, h), dim = 1) if x is not None else h
        z = self.linear(z)

        if context is not None:
            # All gates get the same transformation
            a, b = context.chunk(chunks=2, axis=1)
            z = z * (1 + a) + b 
        
        i, f, o, g = z.chunk(chunks = 4, axis = 1)

        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(self.output_norm(c))
        return h, c

    def _chrono_init_(self, T_max):
        '''
        Bias initialisation based on: https://arxiv.org/pdf/1804.11188.pdf
        :param T_max: The largest time scale we want to capture
        '''
        b = self.linear[0].bias
        h = len(b) // 4
        b.data.fill_(0)
        b.data[h:2*h] = torch.log(nn.init.uniform_(b.data[h:2*h], 1, T_max - 1))
        b.data[:h] = - b.data[h:2*h]
    

class TailEnsemble(nn.Module):
    '''Multihead linear layer.
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        num_tails (int): Number of heads. Default: 2
    '''
    def __init__(self, input_dim: int, output_dim: int, num_tails: int = 2):
        super().__init__()
        self.tails = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_tails)])
    
    def forward(self, x: torch.Tensor):
        ''' Forward pass of the model.

        Args:
            x: input tensor (batch_size, n_feat)

        Returns:
            output tensor (batch_size, num_tails, n_feat)
        '''
        return torch.stack([tail(x) for tail in self.tails], dim = 1)
    

class EncDecLSTM(nn.Module):
    '''Encoder-Decoder LSTM with FiLM layer for conditioning.

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden features. Default: 64
        num_layers (int): Number of layers. Default: 1
        num_conditions (int): Number of conditions. Default: -1
        num_tails (int): Number of prediction tails, i.e. ensemble members. Default: 16
        T_max (int): Maximum time scale for chrono init. Default: 5
    '''
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64, 
                 num_layers: int = 1,
                 num_conditions: int = -1,
                 num_tails: int = 16,
                 T_max: int = 5
                 ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tails = num_tails 
        self.num_layers = num_layers
        self.use_film = num_conditions > 0
        self.T_max = T_max

        # Define model
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Linear(input_dim, self.hidden_dim)
        # Encoder
        self.processor['encoder'] = nn.ModuleList(
            [LSTMCell(self.hidden_dim, self.hidden_dim, T_max=self.T_max)
             for i in range(self.num_layers)]
        )
        # Decoder
        dec_lstm = [LSTMCell(0, self.hidden_dim, T_max=self.T_max)]
        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                dec_lstm.append(LSTMCell(self.hidden_dim, self.hidden_dim, T_max=self.T_max))
        self.processor['decoder'] = nn.ModuleList(dec_lstm)
        self.processor['to_data'] = TailEnsemble(
            self.hidden_dim, self.input_dim, num_tails=self.num_tails
        )
        
        # FilM layer
        if self.use_film:
            self.processor['embedding'] = nn.Embedding(
                num_embeddings=num_conditions, embedding_dim=8*self.hidden_dim
            ) 
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)
        

    def forward(self, x: torch.Tensor, context: torch.Tensor=None, future_pred: int=1):
        ''' Forward pass of the model.
        Args:
            x: input tensor (batch_size, history, n_feat)
            context: Monthly conditioning input. Default: None
            future_pred: Number of steps to predict. Only used if no context is given.
        Returns:
            x_pred: Predicted output tensor (batch_size, num_tails, future_pred, n_feat)
        '''
        batch_size, history, n_feat = x.shape
        horizon = context.shape[1] - history if context is not None else future_pred 
        h = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(x.device)] * self.num_layers 
        c = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(x.device)] * self.num_layers 

        # Encoder 
        for t in range(history):
            z = self.processor['to_latent'](x[:, t].flatten(start_dim=1)) 
            u = self.processor['embedding'](context[:, t]) if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor['encoder'][i](z, h[i], c[i], context=u)
                z += h[i]
        
        # Decoder
        x_pred = []
        for t in range(horizon):
            z = None
            u = self.processor['embedding'](context[:, history+t]) if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor['decoder'][i](z, h[i], c[i], context=u)    
                z = h[i] if z is None else z + h[i]

            x_out = self.processor['to_data'](z)
            x_pred.append(x_out)

        x_pred = torch.stack(x_pred, dim=2)
        return x_pred


class ResidualLSTM(nn.Module):
    """LSTM trained for residuals. Here, residuals for a LIM."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64, 
                 num_layers: int = 1,
                 num_conditions: int = -1,
                 T_max: int = 5) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_film = num_conditions > 0
        self.T_max = T_max

        # Define model
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Linear(input_dim, self.hidden_dim)
        # Decoder
        self.processor['decoder'] = nn.ModuleList(
            [LSTMCell(self.hidden_dim, self.hidden_dim, T_max=self.T_max)
             for i in range(self.num_layers)]
        )
        self.processor['to_data'] = nn.Linear(self.hidden_dim, self.input_dim)
        
        # FilM layer
        if self.use_film:
            self.processor['embedding'] = nn.Embedding(
                num_embeddings=num_conditions, embedding_dim=8*self.hidden_dim
            ) 
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)

    def forward(self, lim_ensemble, context=None):
        device = lim_ensemble.device
        batch_size, n_members, n_horiz, n_feat = lim_ensemble.shape
        # Stack batch_size and members for efficient forward path
        stack_size = batch_size * n_members
        x_input = lim_ensemble.view(stack_size, n_horiz, n_feat)
        u = context.repeat_interleave(n_members, dim=0) if context is not None else None

        # Initialize lstm states
        h_t = [torch.zeros(stack_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.num_layers 
        c_t = [torch.zeros(stack_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.num_layers 

        x_pred = []
        for t in range(n_horiz):
            # Input to latent
            x_t = x_input[:, t,:]
            z_t = self.processor['to_latent'](x_t.flatten(start_dim=1))
            # Conditioning
            u_t = self.processor['embedding'](u[:,t]) if u is not None else None 

            # LSTM Decoder
            for j, lstm in enumerate(self.processor['decoder']):
                h_t[j], c_t[j] = lstm(z_t, h_t[j], c_t[j], u_t)
                z_t += h_t[j]

            # To input space
            x_hat = x_t + self.processor['to_data'](z_t)
            x_pred.append(x_hat.unsqueeze(dim=1))

        # transform list to tensor
        x_pred = torch.cat(x_pred, dim=1)

        return x_pred.view(batch_size, n_members, n_horiz, n_feat)