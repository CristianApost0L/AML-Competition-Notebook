import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config

class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)

class ResidualBlock(nn.Module):
    
    """
    A residual block with two linear layers, layer normalization,
    GELU activation, and dropout.
    """

    def __init__(self, dim, dropout_p=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.final_dropout(self.final_activation(x + self.block(x)))

class ResidualMLP(nn.Module):
    """
    A non-linear MLP built by stacking residual blocks (ResNet style).
    """
    def __init__(self, input_dim=config.D_X, output_dim=config.D_Y,
                 hidden_dim=config.HIDDEN_DIM,
                 num_hidden_layers=config.NUM_HIDDEN_LAYERS,
                 dropout_p=config.DROPOUT_P):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.hidden_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_p) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        print(f"ResidualMLP created:")
        print(f"  Input: {input_dim} -> {hidden_dim}")
        print(f"  {num_hidden_layers} x Residual Blocks (dim={hidden_dim})")
        print(f"  Output: {hidden_dim} -> {output_dim}")
        
    def forward(self, x):
        out = self.input_layer(x)
        for block in self.hidden_blocks:
            out = block(out)
        out = self.output_layer(out)
        return out