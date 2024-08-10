'''LSTM models with 2D inputs. 

@Author  :   Jakob Schlör 
@Time    :   2022/09/11 14:47:32
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch
import torch.nn as nn
import functools
from hyblim.model import modules

class LSTMCell(nn.Module):
    '''LSTMCell for 1d inputs.'''
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 condition_dim: int = 0,
                 T_max: int = -1):
        """
        Args:
            input_dim (int): Input dimensions 
            hidden_dim (int): Hidden dimensions
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + condition_dim, hidden_dim * 4),
            nn.GroupNorm(num_channels=4*hidden_dim, num_groups=4)
        )

        if isinstance(T_max, int) and T_max > 1:
            self._chrono_init_(T_max)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        '''LSTM forward pass
        Args:
            x (torch.Tensor): Input
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
            u (torch.Tensor): Condition (not used, just makes implementation easier)
        '''
        z = torch.cat((x, h), dim = 1) if x is not None else h
        z = self.linear(z)

        i, f, o, g = z.chunk(chunks = 4, axis = 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
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


class FiLMLSTMCell(nn.Module):
    '''LSTMCell for 1d inputs with FiLM layer for conditioning.'''
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 T_max: int = -1):
        """
        Args:
            input_dim (int): Input dimensions 
            hidden_dim (int): Hidden dimensions
            condition_dim (int): Condition dimensions
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 4),
            nn.GroupNorm(num_channels=4*hidden_dim, num_groups=4, affine=False)
        )
        self.output_norm = nn.GroupNorm(num_channels=hidden_dim, num_groups=1, affine=True)

        if isinstance(T_max, int) and T_max > 1:
            self._chrono_init_(T_max)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        '''LSTM forward pass
        Args:
            x (torch.Tensor): Input
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
            u (torch.Tensor): Condition 
        '''
        z = torch.cat((x, h), dim = 1) if x is not None else h
        z = self.linear(z)

        if u is not None:
            # Each gate gets a seperate transformation
            # a, b = u.chunk(chunks=2, axis=1)
            # z = z * a + b # or z*(1+a)+b
            # i, f, o, g = z.chunk(chunks=4, axis=1)

            # All gates get the same transformation
            a, b = u.chunk(chunks=2, axis=1)
            z = z * (1 + a) + b # or z*(1+a)+b
        
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
    

class TeacherFocingLSTM(nn.Module):
    """Teacher forcing LSTM module.

    Args:
        input_dim (int, optional): Number of input features. Defaults to 1.
        hidden_layers (int, optional): Hidden layers of MLP. Defaults to 64.
    """
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.linear_input = nn.Linear(input_dim, self.hidden_dim)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(self.hidden_dim, self.hidden_dim) for i in range(self.n_layers)]
        )
        self.linear_output = nn.Linear(self.hidden_dim, input_dim)
    

    def forward(self, x_hist, future_preds=0):
        batch_size, n_times, n_feat = x_hist.shape
        device = x_hist.device
        h_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 
        c_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 

        outputs = []
        for x_t in x_hist.split(1, dim=1):
            # batch_size, n_feat
            z_t = self.linear_input(x_t.flatten(start_dim=1))
            for j, lstmcell in enumerate(self.lstm):
                h_t[j], c_t[j] = lstmcell(z_t, (h_t[j], c_t[j]))
                z_t = h_t[j]
            out_t = self.linear_output(h_t[j])
            outputs.append(out_t.unsqueeze(dim=1))

        for i in range(future_preds):
            z_t = self.linear_input(out_t)
            for j, lstmcell in enumerate(self.lstm):
                h_t[j], c_t[j] = lstmcell(z_t, (h_t[j], c_t[j]))
                z_t = h_t[j]
            out_t = self.linear_output(h_t[j])
            outputs.append(out_t.unsqueeze(dim=1))

        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs
        

class FilmEncDecLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=1,
                 condition_dim=None, members=16,
                 T_max=5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.members = members
        self.n_layers = n_layers
        self.condition_dim = condition_dim

        if self.condition_dim is None:
            lstm_cell = LSTMCell
        else:
            # FiLM
            lstm_cell = FiLMLSTMCell


            # Dictionary embedding
            self.film = nn.Embedding(num_embeddings=self.condition_dim,
                                     embedding_dim=8*self.hidden_dim) 
            nn.init.uniform_(self.film.weight, 0.05, 0.05)


        self.processor = nn.ModuleDict()
        # Input layer
        self.processor['to_latent'] = nn.Linear(input_dim, self.hidden_dim)

        # Encoder
        self.processor['encoder'] = nn.ModuleList(
            [lstm_cell(self.hidden_dim, self.hidden_dim, T_max=T_max)
             for i in range(n_layers)]
        )

        # Decoder
        dec_lstm = [lstm_cell(0, self.hidden_dim, T_max=T_max)]
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                dec_lstm.append(lstm_cell(self.hidden_dim, self.hidden_dim, T_max=T_max))
        self.processor['decoder'] = nn.ModuleList(dec_lstm)

        # Output layer
        self.processor['to_data'] = modules.MultiHeadLinear(
            self.hidden_dim, self.input_dim, num_heads=self.members)
        

    def forward(self, x_hist, u_hist=None, u_horiz=None, future_preds=0):
        batch_size, n_times, n_feat = x_hist.shape
        device = x_hist.device
        h_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 
        c_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 

        # Encoding
        for t, x_t in enumerate(x_hist.split(1, dim=1)):
            # batch_size, n_feat
            z_t = self.processor['to_latent'](x_t.flatten(start_dim=1))
            u_t = u_hist[:, t] if u_hist is not None else None
            if self.condition_dim is not None:
                u_t = self.film(u_t)
            for j, lstm in enumerate(self.processor['encoder']):
                h_t[j], c_t[j] = lstm(z_t, h_t[j], c_t[j], u_t)
                z_t += h_t[j]

        # Decoding
        x_out = []
        n_rollout = u_horiz.shape[1] if u_horiz is not None else future_preds
        for t in range(n_rollout):
            u_t = u_horiz[:, t] if u_horiz is not None else None
            if self.condition_dim is not None:
                u_t = self.film(u_t)
            z_t = None
            for j, lstm in enumerate(self.processor['decoder']):
                h_t[j], c_t[j] = lstm(z_t, h_t[j], c_t[j], u_t)
                z_t = h_t[j] if z_t is None else z_t + h_t[j]

            # To data space
            x_t = self.processor['to_data'](z_t)
            x_out.append(x_t.unsqueeze(dim=2))

        # transform list to tensor
        x_out = torch.cat(x_out, dim=2)

        return x_out 


class ResidualLSTM(nn.Module):
    """LSTM trained for residuals. Here, residuals for a LIM."""
    def __init__(self, input_dim: int=1, hidden_dim: int=64, n_layers: int=1, 
                 condition_dim: int = None,
                 T_max: int=5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.condition_dim = condition_dim

        # Conditioning
        if self.condition_dim is None:
            lstm_cell = LSTMCell
        else:
            # FiLM
            lstm_cell = FiLMLSTMCell

            # One-hot conditioning
            # cond_layer_scale=16
            # self.film = nn.Sequential(
            #     nn.Linear(condition_dim, hidden_dim * cond_layer_scale),
            #     nn.ReLU(),
            #     nn.Linear(cond_layer_scale * hidden_dim, hidden_dim*8)
            # )

            # Dictionary embedding
            self.film = nn.Embedding(num_embeddings=self.condition_dim,
                                     embedding_dim=8*self.hidden_dim) 
            nn.init.uniform_(self.film.weight, 0.05, 0.05)

        self.processor = nn.ModuleDict()
        # Input layer 
        self.processor['to_latent'] = nn.Linear(self.input_dim, self.hidden_dim)

        # Decoder
        self.processor['decoder'] = nn.ModuleList(
            [lstm_cell(self.hidden_dim, self.hidden_dim, T_max=T_max)
             for i in range(n_layers)]
        )
        self.processor['to_data'] = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, lim_ensemble, u=None):
        device = lim_ensemble.device
        batch_size, n_members, n_horiz, n_feat = lim_ensemble.shape
        # Stack batch_size and members for efficient forward path
        stack_size = batch_size * n_members
        x_in = lim_ensemble.view(stack_size, n_horiz, n_feat)
        u_in = u.repeat_interleave(n_members, dim=0) if u is not None else None

        # Initialize lstm states
        h_t = [torch.zeros(stack_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 
        c_t = [torch.zeros(stack_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 

        x_out = []
        for t in range(n_horiz):
            # Input to latent
            x_t = x_in[:, t,:]
            z_t = self.processor['to_latent'](x_t.flatten(start_dim=1))
            # Conditioning
            u_t = self.film(u_in[:,t]) if u_in is not None else None 

            # LSTM Decoder
            for j, lstm in enumerate(self.processor['decoder']):
                h_t[j], c_t[j] = lstm(z_t, h_t[j], c_t[j], u_t)
                z_t += h_t[j]

            # To input space
            x_hat_t = x_t + self.processor['to_data'](z_t)
            x_out.append(x_hat_t.unsqueeze(dim=1))

        # transform list to tensor
        x_out = torch.cat(x_out, dim=1)

        # Reshape back
        x_out = x_out.view(batch_size, n_members, n_horiz, n_feat)

        return x_out 
            

class ResidualLSTM_dist2ens(nn.Module):
    """LSTM trained for residuals. Here, residuals for a LIM."""
    def __init__(self, input_dim: int=1, hidden_dim: int=64, n_layers: int=1, 
                 condition_dim: int = None, members: int = 16,
                 T_max: int=5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.condition_dim = condition_dim
        self.members = members

        # Conditioning
        if self.condition_dim is None:
            lstm_cell = LSTMCell
        else:
            # FiLM
            lstm_cell = FiLMLSTMCell

            # Dictionary embedding
            self.film = nn.Embedding(num_embeddings=self.condition_dim,
                                     embedding_dim=8*self.hidden_dim) 
            nn.init.uniform_(self.film.weight, 0.05, 0.05)

        self.processor = nn.ModuleDict()
        # Input layer 
        self.processor['to_latent'] = nn.Linear(self.input_dim * 2, self.hidden_dim)

        # Decoder
        self.processor['decoder'] = nn.ModuleList(
            [lstm_cell(self.hidden_dim, self.hidden_dim, T_max=T_max)
             for i in range(n_layers)]
        )
        self.processor['to_data'] = modules.MultiHeadLinear(
            self.hidden_dim, self.input_dim, num_heads=self.members
        ) 

    def forward(self, lim_ensemble, u=None):
        device = lim_ensemble.device
        batch_size, n_members, n_horiz, n_feat = lim_ensemble.shape
        # Stack batch_size and members for efficient forward path
        x_in = torch.stack( 
            (lim_ensemble.mean(dim=1), lim_ensemble.std(dim=1)), dim=-1
        )
        # Initialize lstm states
        h_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 
        c_t = [torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(device)] * self.n_layers 

        x_out = []
        for t in range(n_horiz):
            # Input to latent
            x_t = x_in[:, t,:]
            z_t = self.processor['to_latent'](x_t.flatten(start_dim=1))
            # Conditioning
            u_t = self.film(u[:,t]) if u is not None else None 

            # LSTM Decoder
            for j, lstm in enumerate(self.processor['decoder']):
                h_t[j], c_t[j] = lstm(z_t, h_t[j], c_t[j], u_t)
                z_t += h_t[j]

            # To input space
            x_lim_t = lim_ensemble[:, :, t, :]
            x_hat_t = x_lim_t + self.processor['to_data'](z_t)
            x_out.append(x_hat_t.unsqueeze(dim=2))

        # transform list to tensor
        x_out = torch.cat(x_out, dim=2)

        return x_out 
        