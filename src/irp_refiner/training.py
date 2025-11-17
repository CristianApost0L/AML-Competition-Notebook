import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR

def train_model(model, train_loader, val_loader, device, epochs, lr,
                     save_path, patience=10, min_delta=1e-5, resume=True):
    
    initial_value = torch.ones([], device=device) * np.log(1 / 0.07)
    logit_scale = nn.Parameter(initial_value)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        list(model.parameters()) + [logit_scale],
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    warmup_epochs = min(10, epochs // 4)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [warmup_epochs])
    plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    tqdm_every = 5
    
    if resume and Path(save_path).exists():
        try:
            print(f"Loading checkpoint from {save_path}...")
            checkpoint = torch.load(save_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            logit_scale.data = checkpoint['logit_scale'].data
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'plateau_scheduler_state_dict' in checkpoint:
                plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            
            print(f"   ✓ Resuming from epoch {start_epoch}")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch...")
            start_epoch = 0

    print("Showing only every", tqdm_every, "epochs.")

    for epoch in range(start_epoch, epochs):
        # Enable tqdm only every N epochs
        use_tqdm = (epoch % tqdm_every == 0)
        data_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False) if use_tqdm else train_loader
        
        model.train()
        train_loss = 0

        for X_batch, y_batch in data_iter:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            outputs_norm = outputs
            target_norm = y_batch

            logit_scale_exp = logit_scale.exp().clamp(max=100)
            logits_per_output = logit_scale_exp * torch.matmul(outputs_norm, target_norm.t())
            logits_per_target = logits_per_output.t()

            N = outputs.size(0)
            labels = torch.arange(N, dtype=torch.long).to(device)

            loss_outputs = loss_fn(logits_per_output, labels)
            loss_targets = loss_fn(logits_per_target, labels)
            loss = (loss_outputs + loss_targets) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            if use_tqdm:
                data_iter.set_postfix(loss=f"{loss.item():.4f}", T=f"{logit_scale_exp.item():.2f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                outputs_norm = outputs
                target_norm = y_batch

                logit_scale_exp = logit_scale.exp().clamp(max=100)
                logits_per_output = logit_scale_exp * torch.matmul(outputs_norm, target_norm.t())
                logits_per_target = logits_per_output.t()

                N = outputs.size(0)
                labels = torch.arange(N, dtype=torch.long).to(device)

                loss_outputs = loss_fn(logits_per_output, labels)
                loss_targets = loss_fn(logits_per_target, labels)
                loss = (loss_outputs + loss_targets) / 2
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        if epoch < warmup_epochs or epoch % 5 != 0:
            scheduler.step()
        else:
            plateau_scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LogitT = {logit_scale_exp.item():.4f}, LR = {current_lr:.2e}")
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'logit_scale': logit_scale,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, save_path)
            print(f"   ✓ Saved 'best' model (val_loss={val_loss:.6f}) to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"   (No improvement: {epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\\n--- Early Stopping ---")
            break

    print(f"\\nTraining completed. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss