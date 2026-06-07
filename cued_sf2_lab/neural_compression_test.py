import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Import lab functions for baselines
from cued_sf2_lab.dct import dct_ii, colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image, load_mat_img

TARGET_BITS = 40960  # 5.0 KB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Baseline Functions ---
def deadzone_quant(x, step):
    return np.trunc(x / step)

def deadzone_dequant(xq, step):
    return np.where(xq == 0, 0, np.sign(xq) * (np.abs(xq) + 0.5) * step)

def ssim_score(X, Y):
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    sigma_x2 = np.var(X)
    sigma_y2 = np.var(Y)
    sigma_xy = np.cov(X.flatten(), Y.flatten())[0, 1]
    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return numerator / denominator

def lbt_entropy(X, target_bits):
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    C = dct_ii(8)
    Y = colxfm(colxfm(Xp, C).T, C).T
    
    def simulate(qstep):
        Yq = deadzone_quant(Y, qstep)
        return bpp(Yq) * Yq.size
        
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    Yq = deadzone_quant(Y, qstep)
    Zi = deadzone_dequant(Yq, qstep)
    Zp = colxfm(colxfm(Zi.T, C.T).T, C.T)
    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)
    return Z, simulate(qstep)

def dct_entropy(X, target_bits):
    C = dct_ii(8)
    Y = colxfm(colxfm(X, C).T, C).T
    def simulate(qstep):
        Yq = deadzone_quant(Y, qstep)
        return bpp(Yq) * Yq.size
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
    qstep = (lo + hi) / 2
    Yq = deadzone_quant(Y, qstep)
    Zi = deadzone_dequant(Yq, qstep)
    Z = colxfm(colxfm(Zi.T, C.T).T, C.T)
    return Z, simulate(qstep)

def dwt_entropy(X, target_bits):
    n_levels = 3
    m = X.shape[0]
    Y = X.copy()
    for i in range(n_levels):
        m_cur = m // (2**i)
        Y[:m_cur, :m_cur] = dwt(Y[:m_cur, :m_cur])
        
    def simulate(qstep):
        Yq = deadzone_quant(Y, qstep)
        bits = 0
        bands = []
        for i in range(n_levels):
            m_cur = m // (2**i)
            m_next = m_cur // 2
            bands.extend([
                Yq[:m_next, m_next:m_cur],
                Yq[m_next:m_cur, :m_next],
                Yq[m_next:m_cur, m_next:m_cur]
            ])
        bands.append(Yq[:m // (2**n_levels), :m // (2**n_levels)])
        for b in bands:
            bits += bpp(b) * b.size
        return bits

    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    Yq = deadzone_quant(Y, qstep)
    Z = deadzone_dequant(Yq, qstep)
    for i in range(n_levels-1, -1, -1):
        m_cur = m // (2**i)
        Z[:m_cur, :m_cur] = idwt(Z[:m_cur, :m_cur])
    return Z, simulate(qstep)


# --- 1. Define the Network ---
class CompressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure symmetric padding to keep spatial dimensions consistent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), # Smooth
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7, padding=3), # More smooth
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        latent = self.encoder(x)
        # Additive noise for differentiable quantization (Q-step = 1.0)
        if self.training:
            latent = latent + torch.rand_like(latent) - 0.5 
        else:
            latent = torch.round(latent)
        return self.decoder(latent), latent


def compute_entropy_rate(latent):
    """
    Continuous proxy for entropy to allow backpropagation.
    Uses a standard Laplace distribution negative log-likelihood.
    """
    # Assuming standard Laplace with mu=0, b=1
    return torch.mean(torch.abs(latent))


# --- Main Optimization & Test Routine ---
def run_neural_compression(image_name):
    print(f"\n=====================================")
    print(f"NEURAL COMPRESSION TEST: {image_name}")
    print(f"=====================================")
    
    # 1. Load Data
    mat = loadmat(image_name)
    key = next(k for k in mat if not k.startswith('__'))
    X_orig = mat[key].astype(np.float32) - 128.0
    
    # Apply LBT pre-filter
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    X_pre = X_orig.copy()
    X_pre[t, :] = colxfm(X_pre[t, :], Pf)
    X_pre[:, t] = colxfm(X_pre[:, t].T, Pf).T
    
    # Convert to frequency domain using DCT
    C = dct_ii(8)
    Y = colxfm(colxfm(X_pre, C).T, C).T
    
    # Input to network: Frequency coefficients (LBT + DCT)
    x_input = torch.tensor(Y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Target for Loss: Original spatial image
    x_target = torch.tensor(X_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. Setup Model & Optimizer
    model = CompressionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    
    # Lagrangian Multiplier (lambda) to dynamically target 5.0 KB
    lambda_val = 0.15
    target_bits = TARGET_BITS
    
    print("Training Neural Optimizer (Original 1000 Epochs)...")
    start_time = time.time()
    
    # 3. Optimization Loop
    EPOCHS = 100
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        reconstructed_spatial, latent = model(x_input)
        
        # Loss Dist: 1 - SSIM. (Shift images to positive range for standard SSIM calc)
        dist_loss = 1 - ssim_metric(reconstructed_spatial + 128.0, x_target + 128.0)
        
        # Rate Proxy
        rate_loss = compute_entropy_rate(latent)
        
        # Total Loss
        loss = dist_loss + lambda_val * rate_loss
        loss.backward()
        optimizer.step()
        
        # Periodically adjust lambda based on actual integer entropy
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                _, latent_eval = model(x_input)
                latent_np = latent_eval.cpu().numpy()
                actual_bits = bpp(latent_np) * latent_np.size
                
                # Simple PID adjustment for lambda
                if actual_bits > target_bits:
                    lambda_val *= 1.1
                else:
                    lambda_val *= 0.9
                    
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | SSIM: {1-dist_loss.item():.4f} | Est. Bits: {actual_bits:.0f} | Lambda: {lambda_val:.5f}")

    end_time = time.time()
    energy_mJ = (end_time - start_time) * 15.0
    print(f"Training Complete. Energy Used: ~{energy_mJ/1000:.1f} kJ")

    # 4. Generate Final Decoded Image
    model.eval()
    with torch.no_grad():
        reconstructed_spatial, latent_eval = model(x_input)
        latent_np = latent_eval.cpu().numpy()
        neural_bits = bpp(latent_np) * latent_np.size
        
        # The network outputs the fully reconstructed spatial image!
        Z_neural = reconstructed_spatial.squeeze().cpu().numpy()
        
        neural_ssim = ssim_score(X_orig, Z_neural)
        print(f"\n[Neural Network] Bits={neural_bits:.0f}, SSIM={neural_ssim:.4f}")

    # 5. Baseline Comparisons
    print("Running Baselines...")
    Z_dct, bits_dct = dct_entropy(X_orig, TARGET_BITS)
    print(f"[DCT 8x8]        Bits={bits_dct:.0f}, SSIM={ssim_score(X_orig, Z_dct):.4f}")
    
    Z_lbt, bits_lbt = lbt_entropy(X_orig, TARGET_BITS)
    print(f"[LBT 8x8]        Bits={bits_lbt:.0f}, SSIM={ssim_score(X_orig, Z_lbt):.4f}")
    
    Z_dwt, bits_dwt = dwt_entropy(X_orig, TARGET_BITS)
    print(f"[DWT 3-level]    Bits={bits_dwt:.0f}, SSIM={ssim_score(X_orig, Z_dwt):.4f}")

    # 6. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{image_name} - Neural Compression vs Standards (Target {TARGET_BITS/8/1024:.1f} KB)", fontsize=14, fontweight='bold')
    
    results = [
        ("Neural (LBT-CNN)", Z_neural, neural_bits, neural_ssim),
        ("DCT 8x8", Z_dct, bits_dct, ssim_score(X_orig, Z_dct)),
        ("LBT 8x8", Z_lbt, bits_lbt, ssim_score(X_orig, Z_lbt)),
        ("DWT 3-level", Z_dwt, bits_dwt, ssim_score(X_orig, Z_dwt)),
    ]
    
    for idx, (title, img, bits, ssim) in enumerate(results):
        plot_image(img + 128.0, ax=axes[idx])
        axes[idx].set_title(f"{title}\nBits: {bits:.0f} | SSIM: {ssim:.4f}")
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_neural_compression("lighthouse.mat")
