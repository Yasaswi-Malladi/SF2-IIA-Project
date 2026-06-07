import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL_TARGET_BITS = 40960  # 5.0 KB

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

class C3Advanced(nn.Module):
    def __init__(self, out_shape=(256, 256)):
        super().__init__()
        self.out_shape = out_shape
        
        # 8 hierarchical grids, skipping 256
        # Resolutions: 1, 2, 4, 8, 16, 32, 64, 128
        self.grids = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, 1, 1)),
            nn.Parameter(torch.zeros(1, 1, 2, 2)),
            nn.Parameter(torch.zeros(1, 1, 4, 4)),
            nn.Parameter(torch.zeros(1, 1, 8, 8)),
            nn.Parameter(torch.zeros(1, 1, 16, 16)),
            nn.Parameter(torch.zeros(1, 1, 32, 32)),
            nn.Parameter(torch.zeros(1, 1, 64, 64)),
            nn.Parameter(torch.zeros(1, 1, 128, 128))
        ])
        
        # Deeper Synthesis MLP
        # 8 grids concatenated -> 8 channels
        self.synthesis = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(12, 12, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(12, 1, kernel_size=1)
        )
        
        # Initialize MLP to output 0
        nn.init.zeros_(self.synthesis[-1].weight)
        nn.init.zeros_(self.synthesis[-1].bias)

    def get_quantized_latents(self):
        latents = []
        for g in self.grids:
            if self.training:
                noise = torch.rand_like(g) - 0.5
                latents.append(g + noise)
            else:
                latents.append(torch.round(g))
        return latents

    def forward(self):
        latents = self.get_quantized_latents()
        
        upsampled = []
        for lat in latents:
            up = F.interpolate(lat, size=self.out_shape, mode='bilinear', align_corners=False)
            upsampled.append(up)
            
        x = torch.cat(upsampled, dim=1) # Shape: (1, 8, 256, 256)
        out = self.synthesis(x)
        return out, latents

def compute_entropy_rate(latents):
    total_rate = 0
    for lat in latents:
        mu = lat.mean()
        b = (lat - mu).abs().mean() + 1e-5
        prob = 1.0 / (2 * b) * torch.exp(-(lat - mu).abs() / b)
        prob = torch.clamp(prob, min=1e-10)
        total_rate += -torch.log2(prob).mean() * lat.numel()
    return total_rate

def get_mlp_bits(model):
    params = sum(p.numel() for p in model.synthesis.parameters())
    return params * 16

def run_c3_advanced_optimization():
    print("Loading Image...")
    mat = loadmat("lighthouse.mat")
    key = next(k for k in mat if not k.startswith('__'))
    X_orig = mat[key].astype(np.float32) - 128.0
    
    t_orig = torch.tensor(X_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    mse_loss = nn.MSELoss()
    
    model = C3Advanced(out_shape=(256, 256)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    beta = 0.01 
    
    print(f"C3 Advanced Synthesis Params: {sum(p.numel() for p in model.synthesis.parameters())}")
    mlp_overhead = get_mlp_bits(model)
    print(f"Overhead Bits (MLP): {mlp_overhead}")
    
    print("\nStarting Phase 1: Warmup (Lambda = 0)...")
    PHASE1_EPOCHS = 1000
    start_time = time.time()
    
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        X_hat, latents = model()
        
        dist_loss = 1 - ssim_metric(X_hat + 128.0, t_orig + 128.0)
        fid_loss = mse_loss(X_hat, t_orig)
        
        loss = dist_loss + beta * fid_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == PHASE1_EPOCHS - 1:
            print(f"   Epoch {epoch:4d} | Train SSIM: {1-dist_loss.item():.4f}")
            
    print("\nStarting Phase 2: Rate Squeeze...")
    PHASE2_EPOCHS = 2000
    lambda_val = 0.00005
    
    for epoch in range(PHASE2_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        X_hat, latents = model()
        
        dist_loss = 1 - ssim_metric(X_hat + 128.0, t_orig + 128.0)
        fid_loss = mse_loss(X_hat, t_orig)
        rate_loss = compute_entropy_rate(latents)
        
        loss = dist_loss + beta * fid_loss + lambda_val * rate_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == PHASE2_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                _, latents_eval = model()
                latent_bits = 0
                for lat in latents_eval:
                    latent_np = lat.cpu().numpy()
                    latent_bits += bpp(latent_np) * latent_np.size
                    
                total_bits = latent_bits + mlp_overhead
                
                # PID Loop
                if total_bits > TOTAL_TARGET_BITS:
                    lambda_val *= 1.05
                else:
                    lambda_val *= 0.60 # Drop quickly if overcompressed
                    
            if epoch % 200 == 0 or epoch == PHASE2_EPOCHS - 1:
                print(f"   Epoch {epoch:4d} | SSIM: {1-dist_loss.item():.4f} | Total Bits: {total_bits:.0f} | Lambda: {lambda_val:.6f}")
                
    end_time = time.time()
    print(f"\nOptimization Complete! Energy: ~{(end_time - start_time)*15/1000:.1f} kJ")
    
    # Final Evaluate
    model.eval()
    with torch.no_grad():
        X_hat, latents_eval = model()
        latent_bits = 0
        for lat in latents_eval:
            latent_np = lat.cpu().numpy()
            latent_bits += bpp(latent_np) * latent_np.size
        
        total_bits = latent_bits + mlp_overhead
        eval_ssim = ssim_metric(X_hat + 128.0, t_orig + 128.0).item()
        X_hat_np = X_hat.squeeze().cpu().numpy()

    print(f"\n[Advanced C3 Neural] Total Bits={total_bits:.0f}, SSIM={eval_ssim:.4f}")
    
    # --- Compare with Classical Baselines ---
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("hct", "cued_sf2_lab/huffman_comparison_test.py")
    hct = importlib.util.module_from_spec(spec)
    sys.modules["hct"] = hct
    spec.loader.exec_module(hct)
    
    print("\nRunning Classical Huffman Baselines for Comparison (target 40960 bits)...")
    dct_Z, dct_bits, _ = hct.dct_huffman(X_orig, TOTAL_TARGET_BITS)
    dct_ssim = ssim_score(X_orig, dct_Z)
    
    lbt_Z, lbt_bits, _ = hct.lbt_huffman(X_orig, TOTAL_TARGET_BITS)
    lbt_ssim = ssim_score(X_orig, lbt_Z)
    
    qt_Z, qt_bits, _ = hct.qt_huffman(X_orig, TOTAL_TARGET_BITS)
    # Apply 4x4 Adaptive smoothing to the QT output like we did in earlier tests
    from cued_sf2_lab.huffman_comparison_test import block_adaptive_smoothing
    # Wait, qt_huffman already returns Z. Let's just smooth it.
    qt_Z_smooth = hct.block_adaptive_smoothing(qt_Z, 10.0, block_size=4, strength=1.0)
    qt_ssim = ssim_score(X_orig, qt_Z_smooth)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Neural vs Classical Methods (Target: {TOTAL_TARGET_BITS/8192:.1f} KB)")
    
    plot_image(X_hat_np + 128.0, ax=axes[0, 0])
    axes[0, 0].set_title(f"Advanced C3\nSize: {total_bits/8192:.2f} KB | SSIM: {eval_ssim:.4f}")
    
    plot_image(lbt_Z + 128.0, ax=axes[0, 1])
    axes[0, 1].set_title(f"LBT 8x8 (Huffman)\nSize: {lbt_bits/8192:.2f} KB | SSIM: {lbt_ssim:.4f}")
    
    plot_image(qt_Z_smooth + 128.0, ax=axes[1, 0])
    axes[1, 0].set_title(f"Quadtree (Huffman + Smooth)\nSize: {qt_bits/8192:.2f} KB | SSIM: {qt_ssim:.4f}")
    
    plot_image(dct_Z + 128.0, ax=axes[1, 1])
    axes[1, 1].set_title(f"DCT 8x8 (Huffman)\nSize: {dct_bits/8192:.2f} KB | SSIM: {dct_ssim:.4f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_c3_advanced_optimization()
