import sys
import os
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.baseline_utils import load_data, generate_submission
from src.models.mlp_direct import ResidualMLP_BN, SwiGLUMLP, ModernSwiGLU
from src.ensembling import DirectEnsembleWrapper
from src.evaluation import MLPWrapper

def main():
    parser = argparse.ArgumentParser(description="Generate submission for Direct Models.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True, 
        choices=['I', 'C', 'Modern', 'E'],
        help="Type of model to use for submission (I, C, Modern, E)."
    )
    args = parser.parse_args()

    DEVICE = config.DEVICE
    print(f"--- Generate Submission with Model: {args.model_type} ---")
    
    models = []
    if args.model_type == 'I':
        model = ResidualMLP_BN(num_layers=2, dropout=0.4).to(DEVICE)
        model.load_state_dict(torch.load("model_I_ResidualMLP_NoNorm.pth"))
        wrapper_submission = MLPWrapper(model, DEVICE)
        
    elif args.model_type == 'C':
        model = SwiGLUMLP(num_layers=2, dropout=0.4).to(DEVICE)
        model.load_state_dict(torch.load("model_C_SwiGLU_NoNorm.pth"))
        wrapper_submission = MLPWrapper(model, DEVICE)
        
    elif args.model_type == 'Modern':
        for seed in config.MODERN_SEEDS:
            # Assumes models were trained and saved with seeds in their names
            # For simplicity, let's assume they are in memory (this is a limitation of this script)
            # A better way: train_single_modern_model should save its model to a file
            print("Load Modern Models... (not implemented in this script)")
            # This is complex, as they are not saved to disk by default.
            # For a real script, train_single_modern_model should save to "modern_seed_42.pth" etc.
            raise NotImplementedError("Modern model loading needs saved files.")
        wrapper_submission = DirectEnsembleWrapper(models, DEVICE)

    elif args.model_type == 'E':
        m1 = ResidualMLP_BN(num_layers=3, dropout=0.3, hidden_dim=1536).to(DEVICE)
        m1.load_state_dict(torch.load("ensemble_m1.pth"))
        m2 = ResidualMLP_BN(num_layers=2, dropout=0.4, hidden_dim=2048).to(DEVICE)
        m2.load_state_dict(torch.load("ensemble_m2.pth"))
        m3 = SwiGLUMLP(num_layers=2, dropout=0.4, hidden_dim=1536).to(DEVICE)
        m3.load_state_dict(torch.load("ensemble_m3.pth"))
        models = [m1, m2, m3]
        wrapper_submission = DirectEnsembleWrapper(models, DEVICE)

    # --- Carica dati di Test ---
    test_data = load_data(config.TEST_DATA_PATH)
    X_test_np = test_data['captions/embeddings']
    test_ids = test_data['captions/ids']
    print(f"Test data loaded: {len(X_test_np)} samples.")

    # --- Generate Predictions ---
    print("Generate prediction of test data...")
    y_test_pred = wrapper_submission.translate(X_test_np, batch_size=512)

    # --- Save File ---
    submission_filename = f"submission_direct_{args.model_type}.csv"
    generate_submission(test_ids, y_test_pred, submission_filename)
    print(f"✅ Submission saved: {submission_filename}")

if __name__ == "__main__":
    main()