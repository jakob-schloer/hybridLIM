''' Helper modules.

@Author  :   Jakob Schlör 
@Time    :   2023/09/18 15:15:18
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch
import torch.nn as nn


class MultiHeadLinear(nn.Module):
    '''Multihead linear layer'''
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 2):
        '''
        input_dim: input dimension
        output_dim: output dimension
        num_heads: number of heads
        '''
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
    
    def forward(self, x: torch.Tensor):
        '''
        x: input tensor (batch_size, *, input_dim)
        return: output tensor (batch_size, num_heads, *, output_dim)
        '''
        return torch.stack([head(x) for head in self.heads], dim = 1)


class MultiHeadConvTranspose3d(nn.Module):
    '''Multihead ConvTranspose3d layer'''
    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: tuple, stride: tuple = 1, padding: tuple = 0,
                 output_padding: tuple = 0, num_heads: int = 2):
        '''
        input_dim: input dimension
        output_dim: output dimension
        num_heads: number of heads
        '''
        super().__init__()
        self.heads = nn.ModuleList([
            nn.ConvTranspose3d(input_channels, output_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor):
        '''
        x: input tensor (batch_size, *, input_dim)
        return: output tensor (batch_size, num_heads, *, output_dim)
        '''
        return torch.stack([head(x) for head in self.heads], dim = 1)