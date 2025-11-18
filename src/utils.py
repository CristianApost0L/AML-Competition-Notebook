import torch
import numpy as np
import random
from src.models.mlp_direct import ResidualMLP_BN, SwiGLUMLP

def set_seed(seed):
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    print(f"Seed set to {seed} for random, numpy, and torch.")
    return seed_worker

def load_direct_ensemble(device, model_paths):
    models = []

    # ------------------- Model 1 -------------------
    m1 = ResidualMLP_BN(num_layers=3, dropout=0.3, hidden_dim=1536).to(device)
    m1.load_state_dict(torch.load(model_paths[0], map_location=device))
    m1.eval()
    models.append(m1)

    # ------------------- Model 2 -------------------
    m2 = ResidualMLP_BN(num_layers=2, dropout=0.4, hidden_dim=2048).to(device)
    m2.load_state_dict(torch.load(model_paths[1], map_location=device))
    m2.eval()
    models.append(m2)

    # ------------------- Model 3 -------------------
    m3 = SwiGLUMLP(num_layers=2, dropout=0.4, hidden_dim=1536).to(device)
    m3.load_state_dict(torch.load(model_paths[2], map_location=device))
    m3.eval()
    models.append(m3)

    return models