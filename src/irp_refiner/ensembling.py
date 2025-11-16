import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np

# Import from our new structure
from .models.mlp import ResidualMLP
from . import config

class EnsembleWrapper:
    def __init__(self, model_paths, irp_paths, device):
        self.device = device
        self.models = []
        self.irps = []

        for path in model_paths:
            model = ResidualMLP(
                input_dim=config.D_X, output_dim=config.D_Y, 
                hidden_dim=config.HIDDEN_DIM,
                num_hidden_layers=config.NUM_HIDDEN_LAYERS, 
                dropout_p=config.DROPOUT_P
            ).to(self.device)
            
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models.append(model)
        
        for path in irp_paths:
            self.irps.append(joblib.load(path))
        
        print(f"Ensemble loaded with {len(self.models)} models.")

    @torch.inference_mode()
    def translate(self, x_raw_np):
        all_predictions = []
        for i, (irp, model) in enumerate(zip(self.irps, self.models)):
            # Stage 1: IRP
            x_irp_np = irp.translate(x_raw_np)
            x_irp_torch = torch.from_numpy(x_irp_np).float().to(self.device)
            
            # Stage 2: MLP Refiner
            temp_ds = TensorDataset(x_irp_torch)
            temp_loader = DataLoader(temp_ds, batch_size=config.BATCH_SIZE*2, shuffle=False)
            
            preds_fold = []
            for (batch,) in temp_loader:
                preds_fold.append(model(batch[0]))
            
            all_predictions.append(torch.cat(preds_fold, dim=0))

        # Average predictions
        ensemble_preds_raw = torch.stack(all_predictions).mean(dim=0)
        
        # Re-normalize the averaged vector
        ensemble_preds_norm = F.normalize(ensemble_preds_raw, p=2, dim=1)
        
        return ensemble_preds_norm.cpu().numpy()