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

class C3Lite(nn.Module):
    def __init__(self, out_shape=(256, 256)):
        super().__init__()
        self.out_shape = out_shape
        
        # Define hierarchical grids as Parameters
        # Grids of resolutions: 8, 16, 32, 64, 128, 256
        self.grids = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, 8, 8)),
            nn.Parameter(torch.zeros(1, 1, 16, 16)),
            nn.Parameter(torch.zeros(1, 1, 32, 32)),
            nn.Parameter(torch.zeros(1, 1, 64, 64)),
            nn.Parameter(torch.zeros(1, 1, 128, 128)),
            nn.Parameter(torch.zeros(1, 1, 256, 256))
        ])
        
        # Small Synthesis MLP
        # 6 grids concatenated -> 6 channels
        self.synthesis = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def get_quantized_latents(self):
        latents = []
        for g in self.grids:
            if self.training:
                # Add uniform noise for differentiable quantization
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
            
        x = torch.cat(upsampled, dim=1) # Shape: (1, 6, 256, 256)
        out = self.synthesis(x)
        return out, latents

def compute_entropy_rate(latents):
    total_rate = 0
    for lat in latents:
        # Soft bit estimation using Laplace distribution proxy
        mu = lat.mean()
        b = (lat - mu).abs().mean() + 1e-5
        prob = 1.0 / (2 * b) * torch.exp(-(lat - mu).abs() / b)
        prob = torch.clamp(prob, min=1e-10)
        total_rate += -torch.log2(prob).mean() * lat.numel()
    return total_rate

def get_mlp_bits(model):
    # MLP is stored in float16 theoretically
    params = sum(p.numel() for p in model.synthesis.parameters())
    return params * 16

def run_c3_optimization():
    print("Loading Image...")
    mat = loadmat("lighthouse.mat")
    key = next(k for k in mat if not k.startswith('__'))
    X_orig = mat[key].astype(np.float32) - 128.0
    
    t_orig = torch.tensor(X_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    mse_loss = nn.MSELoss()
    
    model = C3Lite(out_shape=(256, 256)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    beta = 0.01 
    
    print(f"C3 Lite Synthesis Params: {sum(p.numel() for p in model.synthesis.parameters())}")
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
    PHASE2_EPOCHS = 1500
    lambda_val = 0.00005  # Start much gentler so we don't immediately crash the bits
    
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
                
                # Faster/safer PID Loop
                if total_bits > TOTAL_TARGET_BITS:
                    lambda_val *= 1.05
                else:
                    lambda_val *= 0.60  # Drop the penalty VERY fast if we overcompress
                    
            if epoch % 100 == 0 or epoch == PHASE2_EPOCHS - 1:
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

    print(f"\n[C3 Neural] Total Bits={total_bits:.0f}, SSIM={eval_ssim:.4f}")
    
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f"C3-Lite Architecture (Theoretical Bits: {total_bits:.0f})")
    
    plot_image(X_hat_np + 128.0, ax=ax)
    ax.set_title(f"SSIM: {ssim_score(X_orig, X_hat_np):.4f} | Size: {total_bits/8192:.2f} KB")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_c3_optimization()
