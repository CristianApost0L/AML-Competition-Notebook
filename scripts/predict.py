import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.irp_refiner import config
from src.irp_refiner.baseline_utils import load_data, generate_submission
from src.irp_refiner.models.mlp import ResidualMLP
from src.irp_refiner.ensembling import EnsembleWrapper

def run_single_model_prediction(fold_num):
    print(f"\n--- Generating Submission (Single Model, Fold {fold_num}) ---")
    
    # 1. Load test data
    test_data = load_data(config.TEST_DATA_PATH)
    test_embds_raw_np = test_data['captions/embeddings']
    print(f"Test data loaded: {test_embds_raw_np.shape[0]} samples")

    # 2. Load IRP translator
    irp_path = f"{config.CHECKPOINT_DIR}irp_translator_fold_{fold_num}.pkl"
    irp_translator = joblib.load(irp_path)
    print("Applying IRP transformation...")
    test_embds_IRP_np = irp_translator.translate(test_embds_raw_np)
    test_embds_IRP_torch = torch.from_numpy(test_embds_IRP_np).float()

    # 3. Load MLP model
    model_path = f"{config.CHECKPOINT_DIR}mlp_fold_{fold_num}.pth"
    model_inf = ResidualMLP(
        input_dim=config.D_X, output_dim=config.D_Y, hidden_dim=config.HIDDEN_DIM,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS, dropout_p=config.DROPOUT_P
    ).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model_inf.load_state_dict(checkpoint['model_state_dict'])
    model_inf.eval()
    print(f"MLP model (Fold {fold_num}) loaded.")

    # 4. Generate predictions
    test_dataset = TensorDataset(test_embds_IRP_torch)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2)
    
    all_pred_embds = []
    with torch.no_grad():
        for (batch_embds,) in tqdm(test_loader, desc=f"Generating predictions (Fold {fold_num})"):
            batch_embds = batch_embds.to(config.DEVICE)
            pred_batch = model_inf(batch_embds).cpu()
            all_pred_embds.append(pred_batch)

    pred_embds = torch.cat(all_pred_embds, dim=0)

    # 5. Generate submission file
    submission_filename = f'submission_fold_{fold_num}.csv'
    generate_submission(test_data['captions/ids'], pred_embds, submission_filename)
    print(f"Submission file '{submission_filename}' generated.")

def run_ensemble_prediction():
    print("\n--- Generating Submission (K-Fold Ensemble) ---")
    
    # 1. Load test data
    test_data = load_data(config.TEST_DATA_PATH)
    test_embds_raw_np = test_data['captions/embeddings']
    print(f"Test data loaded: {test_embds_raw_np.shape[0]} samples")

    # 2. Load Ensemble Wrapper
    model_paths = [f"{config.CHECKPOINT_DIR}mlp_fold_{f}.pth" for f in range(config.K_FOLDS)]
    irp_paths = [f"{config.CHECKPOINT_DIR}irp_translator_fold_{f}.pkl" for f in range(config.K_FOLDS)]
    
    ensemble_wrapper = EnsembleWrapper(model_paths, irp_paths, config.DEVICE)

    # 3. Generate predictions
    print("Applying ensemble pipeline to test data...")
    pred_embds_ensemble = ensemble_wrapper.translate(test_embds_raw_np)

    # 4. Generate submission file
    submission_filename = 'submission_KFold_Ensemble.csv'
    generate_submission(test_data['captions/ids'], pred_embds_ensemble, submission_filename)
    print(f"Submission file '{submission_filename}' generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate submission files.")
    parser.add_argument(
        '--fold', 
        type=int, 
        help="The fold number to use for a single model submission."
    )
    parser.add_argument(
        '--ensemble', 
        action='store_true', 
        help="Use all K-Folds to create an ensemble submission."
    )
    
    args = parser.parse_args()

    if args.ensemble:
        run_ensemble_prediction()
    elif args.fold is not None:
        if 0 <= args.fold < config.K_FOLDS:
            run_single_model_prediction(args.fold)
        else:
            print(f"Error: Fold number must be between 0 and {config.K_FOLDS - 1}")
    else:
        print("Error: You must specify either --ensemble or --fold <number>")
        parser.print_help()

if __name__ == "__main__":
    main()