import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import set_seed
from src.data_processing import load_and_prep_data_direct
from src.models.mlp import ResidualMLP_BN, SwiGLUMLP
from src.training import train_standard_direct, train_single_modern_model, create_direct_ensemble
from src.evaluation import MLPWrapper, evaluate_retrieval_full, aml_inbatch_retrieval
from src.ensembling import DirectEnsembleWrapper

def main():
    worker_init_fn = set_seed(config.SEED)
    DEVICE = config.DEVICE
    print(f"Using device: {DEVICE}")

    # 1. Load Data using Notebook 2's logic
    (X_train, y_train, X_val, y_val, 
     val_text_embd, val_img_embd_unique, val_label_gt) = load_and_prep_data_direct(
        train_path=config.TRAIN_DATA_PATH,
        coco_path=config.MY_DATA_PATH,
        use_coco=config.USE_COCO_DATASET,
        noise_threshold=config.NOISE_THRESHOLD,
        val_split_ratio=config.VAL_SPLIT_RATIO,
        random_seed=config.SEED
    )
    
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)

    # 2. Train all models (from Cell 8)
    print("\\n--- 1. Training Modello I (ResidualMLP_BN, No-Norm) ---")
    model_I = ResidualMLP_BN(num_layers=2, dropout=0.4).to(DEVICE)
    model_I = train_standard_direct(
        model_I, train_dl, val_dl, config.EPOCHS, config.LR, 
        "model_I_ResidualMLP_NoNorm.pth", 10, 
        use_norm_in_loss=False, device=DEVICE
    )

    print("\\n--- 2. Training Modello C (SwiGLU, No-Norm) ---")
    model_C = SwiGLUMLP(num_layers=2, dropout=0.4).to(DEVICE)
    model_C = train_standard_direct(
        model_C, train_dl, val_dl, config.EPOCHS, config.LR, 
        "model_C_SwiGLU_NoNorm.pth", 10, 
        use_norm_in_loss=False, device=DEVICE
    )

    print("\\n--- 3. Training Modello Modern (Ensemble) ---")
    models_Modern = [
        train_single_modern_model(seed, X_train, y_train, X_val, y_val, config.MODERN_HPARAMS, 10, DEVICE)
        for seed in config.MODERN_SEEDS
    ]

    print("\\n--- 4. Training Modello E (Ensemble) ---")
    models_E = create_direct_ensemble(X_train, y_train, X_val, y_val, DEVICE)

    # 3. Evaluate all models (from Cell 9)
    print("\\n--- Avvio Valutazione Comparativa ---")
    wrapper_I = MLPWrapper(model_I, DEVICE)
    wrapper_C = MLPWrapper(model_C, DEVICE)
    wrapper_Modern = DirectEnsembleWrapper(models_Modern, DEVICE)
    wrapper_E = DirectEnsembleWrapper(models_E, DEVICE)

    print("\\n→ Generazione embeddings normalizzati...")
    # Use the full query set (val_text_embd) for evaluation
    # Note: X_val and val_text_embd are the same tensor
    emb_I = wrapper_I.translate(val_text_embd)
    emb_C = wrapper_C.translate(val_text_embd)
    emb_Modern = wrapper_Modern.translate(val_text_embd)
    emb_E = wrapper_E.translate(val_text_embd)

    print("\\n→ Normalizzazione target embeddings...")
    # Use the unique image gallery (val_img_embd_unique)
    gallery_emb_norm = F.normalize(val_img_embd_unique.float().to(DEVICE), p=2, dim=1)
    
    # --- Full Retrieval (N vs M) ---
    print("\\n🔥 VALUTAZIONE FULL RETRIEVAL (CLIP-style)")
    metrics_I_full = evaluate_retrieval_full(emb_I, gallery_emb_norm, val_label_gt, device=DEVICE)
    metrics_C_full = evaluate_retrieval_full(emb_C, gallery_emb_norm, val_label_gt, device=DEVICE)
    metrics_Modern_full = evaluate_retrieval_full(emb_Modern, gallery_emb_norm, val_label_gt, device=DEVICE)
    metrics_E_full = evaluate_retrieval_full(emb_E, gallery_emb_norm, val_label_gt, device=DEVICE)

    # --- In-Batch Retrieval (N vs N) ---
    print("\\n🔥 VALUTAZIONE IN-BATCH (Competizione Style)")
    # Note: In-batch uses the paired image embeddings (y_val), NOT the full gallery
    gt_emb_paired = y_val.float()
    metrics_I_ib = aml_inbatch_retrieval(torch.from_numpy(emb_I), gt_emb_paired, batch_size=256)
    metrics_C_ib = aml_inbatch_retrieval(torch.from_numpy(emb_C), gt_emb_paired, batch_size=256)
    metrics_Modern_ib = aml_inbatch_retrieval(torch.from_numpy(emb_Modern), gt_emb_paired, batch_size=256)
    metrics_E_ib = aml_inbatch_retrieval(torch.from_numpy(emb_E), gt_emb_paired, batch_size=256)

    # 4. Print Report
    print("\\n\\n" + "="*80 + "\\n          📊 RIEPILOGO FINALE: FULL vs IN-BATCH" + "\\n" + "="*80)
    def print_full_and_inbatch(name, full, ib):
        print(f"\\n🔵 {name}")
        print("- FULL RETRIEVAL (N vs M Gallery)")
        print(f"  Recall@1:   {full.get('recall@1', 0):.4f}")
        print(f"  Recall@5:   {full.get('recall@5', 0):.4f}")
        print(f"  MRR:        {full.get('mrr', 0):.4f}")
        print(f"  L2 Dist:    {full.get('l2_dist', 0):.4f}")
        print("- IN-BATCH RETRIEVAL (N vs N)")
        print(f"  Recall@1:   {ib.get('r1', 0):.4f}")
        print(f"  Recall@5:   {ib.get('r5', 0):.4f}")
        print(f"  MRR:        {ib.get('mrr', 0):.4f}")

    print_full_and_inbatch("Modello I (ResidualMLP_BN)", metrics_I_full, metrics_I_ib)
    print_full_and_inbatch("Modello C (SwiGLU)", metrics_C_full, metrics_C_ib)
    print_full_and_inbatch("Modello Modern (Ensemble)", metrics_Modern_full, metrics_Modern_ib)
    print_full_and_inbatch("Modello E (Ensemble)", metrics_E_full, metrics_E_ib)

if __name__ == "__main__":
    main()