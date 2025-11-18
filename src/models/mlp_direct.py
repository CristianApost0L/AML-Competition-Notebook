import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Modello 1: SwiGLU MLP Semplice (Notebook 2, Modello C) ---
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class ResidualBlockSwiGLU(nn.Module):
    def __init__(self, dim, dropout_p=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.LayerNorm(2 * dim), SwiGLU(),
            nn.Dropout(dropout_p), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        return self.final_dropout(self.final_activation(x + self.block(x)))

class SwiGLUMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=1536, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlockSwiGLU(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

# --- Modello 2: Residual MLP (Notebook 2, Modello I) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout_p), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        return self.final_dropout(self.final_activation(x + self.block(x)))

class ResidualMLP_BN(nn.Module):
    """ Uses BatchNorm instead of LayerNorm """
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=1536, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

# --- Modello 3: Modern SwiGLU (Notebook 2, SOTA) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_(); return x.div(keep_prob) * random_tensor

class ResidualBlockModern(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__(); hidden_dim = dim * 2; self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim); self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim); self.drop_path_rate = drop_path
    def forward(self, x):
        residual = x; x = self.norm(x); x = F.silu(self.w1(x)) * self.w2(x); x = self.w3(x)
        if self.drop_path_rate > 0: x = drop_path(x, self.drop_path_rate, self.training)
        return residual + x

class ModernSwiGLU(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=1536, num_layers=4, drop_path_rate=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), RMSNorm(hidden_dim))
        dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.blocks = nn.ModuleList([ResidualBlockModern(hidden_dim, dpr[i]) for i in range(num_layers)])
        self.final_norm = RMSNorm(hidden_dim); self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def zero_init_residuals(self):
        for block in self.blocks: nn.init.zeros_(block.w3.weight); nn.init.zeros_(block.w3.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x); return self.head(x)