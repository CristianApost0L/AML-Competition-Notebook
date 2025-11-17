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

        # Load all MLP models
        for path in model_paths:
            model = ResidualMLP(
                input_dim=config.D_X, output_dim=config.D_Y, hidden_dim=config.HIDDEN_DIM,
                num_hidden_layers=config.NUM_HIDDEN_LAYERS, dropout_p=config.DROPOUT_P
            ).to(self.device)
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models.append(model)
        
        # Load all IRP translators
        for path in irp_paths:
            self.irps.append(joblib.load(path))
        
        print(f"Ensemble loaded with {len(self.models)} models.")

    @torch.inference_mode()
    def translate(self, x_raw_np):
        all_predictions = []
        
        # Ensure input is a 2D numpy array
        if x_raw_np.ndim == 1:
            x_raw_np = x_raw_np.reshape(1, -1)

        # Get a prediction from each model in the ensemble
        for i, (irp, model) in enumerate(zip(self.irps, self.models)):
            # Stage 1: IRP (on CPU with numpy)
            x_irp_np = irp.translate(x_raw_np)
            x_irp_torch = torch.from_numpy(x_irp_np).float()
            
            # Stage 2: MLP Refiner (on GPU with torch)
            temp_ds = TensorDataset(x_irp_torch)
            # Use a non-zero batch size to ensure output is always 2D
            loader_bs = min(512, len(temp_ds))
            if loader_bs == 0: continue # Skip if empty input
            temp_loader = DataLoader(temp_ds, batch_size=loader_bs, shuffle=False)
            
            preds_fold = []
            for batch_tuple in temp_loader:
                # --- FIX: Correctly unpack the batch tensor ---
                # The loader yields a tuple, e.g., (tensor_chunk,)
                batch_tensor = batch_tuple[0].to(self.device)
                preds_fold.append(model(batch_tensor))
            
            # This should now be a list of 2D tensors
            all_predictions.append(torch.cat(preds_fold, dim=0))

        # This check is crucial for debugging
        if not all_predictions:
            # Handle case of empty input
            return np.array([]).reshape(0, config.D_Y)
            
        # Average the predictions from all models
        ensemble_preds_raw = torch.stack(all_predictions).mean(dim=0)
        
        # --- FIX: Ensure tensor is 2D before normalizing ---
        if ensemble_preds_raw.ndim == 1:
            ensemble_preds_raw = ensemble_preds_raw.unsqueeze(0)

        # CRITICAL: Re-normalize the averaged vector
        ensemble_preds_norm = F.normalize(ensemble_preds_raw, p=2, dim=1)
        
        return ensemble_preds_norm.cpu().numpy()