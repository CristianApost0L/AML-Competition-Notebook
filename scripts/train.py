import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
import gc

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import set_seed
from src.data_processing import load_and_clean_data
from src.models.irp import IRPTranslator
from src.models.mlp import ResidualMLP
from src.training import train_irp_refiner

def main():
    print(f"Using device: {config.DEVICE}")
    worker_init_fn = set_seed(config.SEED)

    # 1. Load and Clean Data
    X_train_np_cleaned, Y_train_np_cleaned = load_and_clean_data(
        config.TRAIN_DATA_PATH, config.NOISE_THRESHOLD
    )

    # 2. Initialize KFold
    kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)

    # 3. K-Fold Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np_cleaned)):
        print("\\n" + "="*80)
        print(f"=============== FOLD {fold+1}/{config.K_FOLDS} ===============")
        print("="*80)

        # --- Split data for this fold ---
        X_train_fold, X_val_fold = X_train_np_cleaned[train_idx], X_train_np_cleaned[val_idx]
        Y_train_fold, Y_val_fold = Y_train_np_cleaned[train_idx], Y_train_np_cleaned[val_idx]

        # --- IRP Stage ---
        print(f"--- FOLD {fold+1}: IRP Stage ---")
        anchor_indices = np.random.choice(len(X_train_fold), config.K_ANCHORS, replace=False)
        X_anchor = X_train_fold[anchor_indices]
        Y_anchor = Y_train_fold[anchor_indices]

        scaler_X = StandardScaler().fit(X_anchor)
        scaler_Y = StandardScaler().fit(Y_anchor)

        irp_translator_fold = IRPTranslator(
            scaler_X, scaler_Y, 
            omega=config.IRP_OMEGA, delta=config.IRP_DELTA, 
            ridge=config.IRP_RIDGE, verbose=False
        )
        irp_translator_fold.fit(X_anchor, Y_anchor)
        print(f"   ✓ IRP translator for fold {fold+1} fitted.")

        irp_path = f"{config.CHECKPOINT_DIR}irp_translator_fold_{fold}.pkl"
        joblib.dump(irp_translator_fold, irp_path)
        print(f"   ✓ IRP translator saved to {irp_path}")

        X_train_IRP_fold = torch.from_numpy(irp_translator_fold.translate(X_train_fold)).float()
        X_val_IRP_fold = torch.from_numpy(irp_translator_fold.translate(X_val_fold)).float()
        print(f"   ✓ Train and Val data transformed for fold {fold+1}.")

        # --- DataLoader Stage ---
        train_ds_fold = TensorDataset(X_train_IRP_fold, torch.from_numpy(Y_train_fold).float())
        val_ds_fold = TensorDataset(X_val_IRP_fold, torch.from_numpy(Y_val_fold).float())

        train_loader_fold = DataLoader(train_ds_fold, batch_size=config.BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
        val_loader_fold = DataLoader(val_ds_fold, batch_size=config.BATCH_SIZE, shuffle=False)

        # --- Model Training Stage ---
        print(f"--- FOLD {fold+1}: MLP Refiner Training Stage ---")
        model_fold = ResidualMLP(
            input_dim=config.D_X, output_dim=config.D_Y, hidden_dim=config.HIDDEN_DIM,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS, dropout_p=config.DROPOUT_P
        ).to(config.DEVICE)

        model_path_fold = f"{config.CHECKPOINT_DIR}mlp_fold_{fold}.pth"

        train_irp_refiner(
            model_fold, train_loader_fold, val_loader_fold, config.DEVICE,
            epochs=config.EPOCHS, lr=config.LR, save_path=model_path_fold,
            patience=config.EARLY_STOP_PATIENCE, min_delta=config.MIN_IMPROVEMENT_DELTA,
            resume=False 
        )

        # --- Clean up memory ---
        del model_fold, train_loader_fold, val_loader_fold, X_train_IRP_fold, X_val_IRP_fold
        gc.collect()
        torch.cuda.empty_cache()

    print("\\n" + "="*80)
    print("K-Fold Training Complete. All models saved.")
    print("="*80)

if __name__ == "__main__":
    main()