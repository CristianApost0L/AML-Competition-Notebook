import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
from . import config
from .models.mlp_direct import ResidualMLP_BN, SwiGLUMLP, ModernSwiGLU

def train_irp_refiner(model, train_loader, val_loader, device, epochs, lr,
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
    best_state = None
    
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
            
            best_state = copy.deepcopy(model.state_dict())
            
            print(f"   Resuming from epoch {start_epoch}")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch...")
            start_epoch = 0

    print("Showing only every", tqdm_every, "epochs.")

    for epoch in range(start_epoch, epochs):
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

        if (epoch + 1) % tqdm_every == 0 or epoch < 5:
            print(f"   Ep {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Temp: {logit_scale_exp.item():.2f} | LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': best_state,
                'logit_scale': logit_scale,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, save_path)
            
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early Stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\\nTraining completed. Best Val Loss: {best_val_loss:.6f}")
    
    if best_state:
        model.load_state_dict(best_state)
    else:
        try:
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: could not load best state at end of training. {e}")
            
    return best_val_loss

def train_standard_direct(model, train_loader, val_loader, epochs, lr, save_path, patience, use_norm_in_loss, device):
    """
    Training function for 'direct' models (ResidualMLP_BN, SwiGLUMLP).
    """
    logit_scale = nn.Parameter(torch.ones([], device=device) * np.log(1 / 0.07))
    optimizer = optim.AdamW(list(model.parameters()) + [logit_scale], lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')
    no_improve = 0
    
    loss_strategy = "Norm-in-Loss" if use_norm_in_loss else "No-Norm-Loss (Raw)"
    print(f"\n--- Start Training Standard Direct ({save_path}) ---")
    print(f"Loss Strategy: {loss_strategy}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_raw = model(bx)
            
            if use_norm_in_loss:
                out = F.normalize(out_raw, p=2, dim=1)
                tgt = F.normalize(by, p=2, dim=1)
            else:
                out = out_raw; tgt = by
            
            scale = logit_scale.exp().clamp(max=100)
            logits = scale * (out @ tgt.T)
            labels = torch.arange(len(out), device=device)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            
            loss.backward(); optimizer.step(); train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out_raw = model(bx)
                if use_norm_in_loss:
                    out = F.normalize(out_raw, p=2, dim=1); tgt = F.normalize(by, p=2, dim=1)
                else:
                    out = out_raw; tgt = by
                scale = logit_scale.exp().clamp(max=100)
                logits = scale * (out @ tgt.T)
                labels = torch.arange(len(out), device=device)
                val_loss += ((loss_fn(logits, labels) + loss_fn(logits.T, labels))/2).item()
        
        avg_val = val_loss / len(val_loader)
        if (epoch+1) % 10 == 0:
             print(f"Ep {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Temp: {scale.item():.2f}")
        scheduler.step(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0; torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early Stopping! (Ep {epoch+1})")
                break

    print(f"Training completed. Best model saved to {save_path}")
    model.load_state_dict(torch.load(save_path))
    return model

def train_single_modern_model(seed, X_tr, y_tr, X_vl, y_vl, hparams, patience, device):
    """
    Training function for the ModernSwiGLU model.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=hparams['batch_size'], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_vl, y_vl), batch_size=hparams['batch_size'], shuffle=False)
    
    model = ModernSwiGLU(
        num_layers=hparams['layers'], hidden_dim=hparams['hidden_dim'],
        drop_path_rate=hparams['drop_path']
    ).to(device)
    model.zero_init_residuals()
    
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), device=device))
    optimizer = optim.AdamW(
        [{'params': model.parameters()}, {'params': [logit_scale], 'weight_decay': 0.0}],
        lr=hparams['lr'], weight_decay=hparams['weight_decay']
    )
    
    warmup = 2
    epochs = hparams['epochs']
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-6)
        ],
        milestones=[warmup]
    )
    
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    
    print(f"\n Training Modern (Seed {seed})")
    print(f"   Config: {hparams['layers']}L, HD={hparams['hidden_dim']}, DP={hparams['drop_path']:.3f}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # NO NORMALIZZAZIONE!
            out = model(bx); tgt = by
            
            scale = logit_scale.exp().clamp(max=100)
            logits = scale * (out @ tgt.T)
            labels = torch.arange(len(out), device=device)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_dl)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                out = model(bx); tgt = by
                scale = logit_scale.exp().clamp(max=100)
                logits = scale * (out @ tgt.T)
                labels = torch.arange(len(out), device=device)
                val_loss += ((loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2).item()
        
        avg_val = val_loss / len(val_dl)
        
        if (epoch + 1) % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Ep {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"Temp: {scale.item():.2f} | LR: {current_lr:.2e}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"   Early Stopping at epoch {epoch+1}")
                break
    
    print(f"   ✔️ Seed {seed} completed | Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_state)
    return model

def create_direct_ensemble(X_train, y_train, X_val, y_val, device):
    """
    Trains the diverse ensemble E.
    """
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)
    models = []
    
    print("\n" + "="*60 + "\nTraining Ensemble Member 1/3: Deep ResidualMLP_BN" + "\n" + "="*60)
    m1 = ResidualMLP_BN(num_layers=3, dropout=0.3, hidden_dim=1536).to(device)
    m1 = train_standard_direct(
        m1, train_dl, val_dl, epochs=100, lr=5e-4,
        save_path="ensemble_m1.pth", patience=10, use_norm_in_loss=False, device=device
    )
    models.append(m1)
    
    print("\n" + "="*60 + "\nTraining Ensemble Member 2/3: Wide ResidualMLP_BN" + "\n" + "="*60)
    m2 = ResidualMLP_BN(num_layers=2, dropout=0.4, hidden_dim=2048).to(device)
    m2 = train_standard_direct(
        m2, train_dl, val_dl, epochs=100, lr=5e-4,
        save_path="ensemble_m2.pth", patience=10, use_norm_in_loss=False, device=device
    )
    models.append(m2)
    
    print("\n" + "="*60 + "\nTraining Ensemble Member 3/3: SwiGLU" + "\n" + "="*60)
    m3 = SwiGLUMLP(num_layers=2, dropout=0.4, hidden_dim=1536).to(device)
    m3 = train_standard_direct(
        m3, train_dl, val_dl, epochs=100, lr=5e-4,
        save_path="ensemble_m3.pth", patience=10, use_norm_in_loss=False, device=device
    )
    models.append(m3)
    
    return models