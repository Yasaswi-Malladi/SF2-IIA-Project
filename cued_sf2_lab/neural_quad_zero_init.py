import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure

from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image

# --- Dynamically import quad-tree test.py ---
spec = importlib.util.spec_from_file_location("quad_tree_test", "cued_sf2_lab/quad-tree test.py")
qt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Targets
TOTAL_TARGET_BITS = 40960  # 5.0 KB
BASE_TARGET_BITS = 32768   # 4.0 KB
NEURAL_BUDGET = TOTAL_TARGET_BITS - BASE_TARGET_BITS # 1.0 KB

def ssim_score(X, Y):
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    sig_x = np.var(X)
    sig_y = np.var(Y)
    sig_xy = np.mean((X - mu_x) * (Y - mu_y))
    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2))

def get_base_quadtree(X, target_bits):
    print("   Searching for Quadtree base Q-step...")
    def simulate(qstep):
        vlc, header, total_bits = qt.qt_encode(X, qstep, var_thresh=0.05)
        return total_bits
        
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    vlc, header, actual_bits = qt.qt_encode(X, qstep, var_thresh=0.05)
    
    print("   Decoding Quadtree image...")
    Z = qt.qt_decode(vlc, header, X.shape)
    
    print("   Applying 4x4 Adaptive Smoothing...")
    Z_smoothed = qt.block_adaptive_smoothing(Z, qstep, block_size=4, strength=1.0, passes=1)
    
    return Z_smoothed, actual_bits

def compute_variance_mask(X_base, block_size=8, var_thresh=50):
    H, W = X_base.shape
    mask = np.zeros_like(X_base)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = X_base[i:i+block_size, j:j+block_size]
            if np.var(block) > var_thresh:
                mask[i:i+block_size, j:j+block_size] = 1.0
    return mask

class ZeroInitResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(16, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvTranspose2d(16, 16, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1, 5), padding=(0, 2)),
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
        )
        
        # --- ZERO-INIT TECHNIQUE ---
        torch.nn.init.zeros_(self.decoder[-1].weight)
        torch.nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, mask):
        x_gated = x * mask
        latent = self.encoder(x_gated)
        if self.training:
            latent = latent + torch.rand_like(latent) - 0.5 
        else:
            latent = torch.round(latent)
        residual_hat = self.decoder(latent)
        return residual_hat * mask, latent

def compute_entropy_rate(latent):
    mu = latent.mean()
    b = (latent - mu).abs().mean() + 1e-5
    prob = 1.0 / (2 * b) * torch.exp(-(latent - mu).abs() / b)
    prob = torch.clamp(prob, min=1e-10)
    return -torch.log2(prob).mean() * latent.numel()

def run_zero_init_optimization():
    print("Loading Image...")
    mat = loadmat("lighthouse.mat")
    key = next(k for k in mat if not k.startswith('__'))
    X_orig = mat[key].astype(np.float32) - 128.0
    
    print("\n1. Computing Quadtree Smoothed Base...")
    X_base, base_bits = get_base_quadtree(X_orig, BASE_TARGET_BITS)
    print(f"   Base Quadtree: {base_bits:.0f} bits | Base SSIM: {ssim_score(X_orig, X_base):.4f}")
    
    print("\n2. Generating Variance Gate Mask...")
    mask_np = compute_variance_mask(X_base, var_thresh=50)
    residual_np = X_orig - X_base
    
    t_orig = torch.tensor(X_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_base = torch.tensor(X_base, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_res = torch.tensor(residual_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    mse_loss = nn.MSELoss()
    
    model = ZeroInitResidualNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    beta = 0.01 
    
    print("\n3. Starting Phase 1: Warmup (Lambda = 0)...")
    PHASE1_EPOCHS = 500
    start_time = time.time()
    
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        res_hat, latent = model(t_res, t_mask)
        X_hat = t_base + res_hat
        
        dist_loss = 1 - ssim_metric(X_hat + 128.0, t_orig + 128.0)
        fid_loss = mse_loss(X_hat, t_orig)
        
        loss = dist_loss + beta * fid_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == PHASE1_EPOCHS - 1:
            print(f"   Epoch {epoch:4d} | Train SSIM: {1-dist_loss.item():.4f}")
            
    print("\n4. Starting Phase 2: Rate Squeeze...")
    PHASE2_EPOCHS = 1000
    lambda_val = 0.001
    
    for epoch in range(PHASE2_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        res_hat, latent = model(t_res, t_mask)
        X_hat = t_base + res_hat
        
        dist_loss = 1 - ssim_metric(X_hat + 128.0, t_orig + 128.0)
        fid_loss = mse_loss(X_hat, t_orig)
        rate_loss = compute_entropy_rate(latent)
        
        loss = dist_loss + beta * fid_loss + lambda_val * rate_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == PHASE2_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                _, latent_eval = model(t_res, t_mask)
                latent_np = latent_eval.cpu().numpy()
                neural_bits = bpp(latent_np) * latent_np.size
                total_bits = base_bits + neural_bits
                
                if total_bits > TOTAL_TARGET_BITS:
                    lambda_val *= 1.15
                else:
                    lambda_val *= 0.90
                    
            if epoch % 200 == 0 or epoch == PHASE2_EPOCHS - 1:
                print(f"   Epoch {epoch:4d} | SSIM: {1-dist_loss.item():.4f} | Total Bits: {total_bits:.0f} | Lambda: {lambda_val:.6f}")
                
    end_time = time.time()
    print(f"\nOptimization Complete! Energy: ~{(end_time - start_time)*15/1000:.1f} kJ")
    
    # Final Evaluate
    model.eval()
    with torch.no_grad():
        res_hat, latent_eval = model(t_res, t_mask)
        latent_np = latent_eval.cpu().numpy()
        neural_bits = bpp(latent_np) * latent_np.size
        total_bits = base_bits + neural_bits
        
        X_hat = t_base + res_hat
        eval_ssim = ssim_metric(X_hat + 128.0, t_orig + 128.0).item()
        X_hat_np = X_hat.squeeze().cpu().numpy()

    print(f"\n[Quadtree Neural Refinement] Total Bits={total_bits:.0f}, SSIM={eval_ssim:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Quadtree + Zero-Init Neural Refinement (Target: 5.0 KB)")
    
    plot_image(X_base + 128.0, ax=axes[0])
    axes[0].set_title(f"Quadtree Base (4x4 Smoothed)\nBits: {base_bits:.0f} | SSIM: {ssim_score(X_orig, X_base):.4f}")
    
    plot_image(X_hat_np + 128.0, ax=axes[1])
    axes[1].set_title(f"Quadtree + Neural Residual\nTotal Bits: {total_bits:.0f} | SSIM: {ssim_score(X_orig, X_hat_np):.4f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_zero_init_optimization()
