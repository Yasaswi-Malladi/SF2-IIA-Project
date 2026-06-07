import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.jpeg import jpegenc, jpegdec

# -------------------------------------------------------------------------
# Mathematical and Coding Theory for Reordered LBT + Quadtree
# -------------------------------------------------------------------------
# 1. Rate-Distortion Theory:
# We aim to minimize the Mean Squared Error (RMS) subject to a bit constraint
# (5.0 KB = 40,960 bits). The LBT concentrates the image energy into the DC
# (low frequency) components, leaving the high-frequency components as mostly zeros.
#
# 2. LBT Overlaps:
# The LBT uses a "pre-filter" (Pf) across the 8x8 block boundaries before the DCT.
# This smears the quantization noise across the edges, preventing the harsh blocking
# artifacts seen in pure DCT-based JPEG compression.
#
# 3. Information Theory (Regrouping and Quadtrees):
# Normally, an LBT/DCT matrix has interleaved frequencies. A massive DC value sits
# right next to a 0-value AC coefficient. A Quadtree cannot compress this because
# Quadtrees look for continuous flat blocks of identical values (zeros).
# By applying `regroup()`, we move all DC components to the top-left, and all
# high frequencies to the bottom-right. This transforms the LBT into a Discrete
# Wavelet Transform (DWT) format! The high-frequency areas become massive, contiguous
# oceans of zeros. The Quadtree can now easily find these 32x32 or 16x16 zero-blocks
# and compress them into a single node!
# -------------------------------------------------------------------------

TOTAL_TARGET_BITS = 40960  # 5.0 KB

def rms_error(X, Y):
    return np.std(X - Y)

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

def deadzone_quant(x, step):
    return np.sign(x) * np.floor(np.abs(x) / step)

def deadzone_dequant(xq, step):
    return xq * step
#

# -------------------------------------------------------------------------
# Quadtree Encoder & Decoder for Reordered Coefficients
# -------------------------------------------------------------------------
def get_qt_structure(Yrq, min_size=4):
    """
    Recursively scans the quantized reordered matrix (Yrq).
    If a block is entirely zeros, it becomes a single leaf node.
    Otherwise, it subdivides until min_size.
    quantisation of the frequency domain, therefore different to spatial quad tree
    """
    blocks = []
    tree_bits = []
    
    def _split(x, y, size):
        block = Yrq[y:y+size, x:x+size]
        
        # If the block is perfectly uniformly zero, it's a flat leaf!
        if np.all(block == 0) or size <= min_size:
            tree_bits.append(0) # 0 means leaf
            blocks.append((x, y, size))
        else:
            tree_bits.append(1) # 1 means split
            half = size // 2
            _split(x, y, half)
            _split(x + half, y, half)
            _split(x, y + half, half)
            _split(x + half, y + half, half)
            
    _split(0, 0, Yrq.shape[0])
    return tree_bits, blocks

def encode_reordered_lbt_qt(X, target_bits):
    """
    Encoder:
    1. LBT Pre-filter
    2. DCT
    3. Regroup (subbands)
    4. Quantize
    5. Quadtree segmentation
    """
    # 1. LBT Pre-filter
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    # 2. DCT 8x8
    C = dct_ii(8)
    Y = colxfm(colxfm(Xp, C).T, C).T
    
    # 3. Regroup into subbands
    Yr = regroup(Y, 8)
    
    def simulate(qstep):
        # 4. Quantize
        Yrq = deadzone_quant(Yr, qstep)
        
        # 5. Quadtree
        tree_bits, blocks = get_qt_structure(Yrq, min_size=4)
        
        # Calculate Entropy
        coef_bits = 0
        for (x, y, size) in blocks:
            block = Yrq[y:y+size, x:x+size]
            if not np.all(block == 0):
                coef_bits += bpp(block) * block.size
                
        total_bits = len(tree_bits) + coef_bits
        return total_bits, Yrq
        
    # Bisection search to hit exactly 5.0 KB
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        bits, _ = simulate(mid)
        if bits > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    final_bits, Yrq = simulate(qstep)
    return Yrq, qstep, final_bits

def decode_reordered_lbt_qt(Yrq, qstep):
    """
    Decoder:
    1. Dequantize
    2. Inverse Regroup
    3. Inverse DCT
    4. Inverse LBT Pre-filter
    """
    # 1. Dequantize
    Yr_hat = deadzone_dequant(Yrq, qstep)
    
    # 2. Inverse Regroup (regrouping the regrouped matrix undoes it)
    Y_hat = regroup(Yr_hat, 256//8) 
    
    # 3. Inverse DCT
    C = dct_ii(8)
    Xp_hat = colxfm(colxfm(Y_hat.T, C.T).T, C.T)
    
    # 4. Inverse LBT
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    X_hat = Xp_hat.copy()
    X_hat[:, t] = colxfm(X_hat[:, t].T, Pr.T).T
    X_hat[t, :] = colxfm(X_hat[t, :], Pr.T)
    
    return X_hat

# -------------------------------------------------------------------------
# Standard LBT Baselines (4x4 and 8x8)
# -------------------------------------------------------------------------
def lbt_entropy(X, target_bits, N=8):
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N//2:-N//2]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    C = dct_ii(N)
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

def lbt_huffman(X, target_bits, N=8):
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N//2:-N//2]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    # Note: jpegenc naturally applies an 8x8 DCT. 
    # If N=4, we would need a custom 4x4 jpegenc, but since jpegenc is hardcoded 
    # for 8x8, we can only truly evaluate 8x8 Huffman safely.
    # For N=4, we will just fallback to Entropy representation if jpegenc fails,
    # but let's try calling it.
    
    if N != 8:
        # Cannot use jpegenc for 4x4 blocks easily as it hardcodes 8x8 DCT.
        # Fallback to pure entropy for 4x4 to avoid crashing.
        return lbt_entropy(X, target_bits, N)
        
    def simulate(qstep):
        vlc, _ = jpegenc(Xp, qstep, log=False)
        return int(np.sum(vlc[:, 1]))
        
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    vlc, _ = jpegenc(Xp, qstep, log=False)
    Zp = jpegdec(vlc, qstep, log=False)
    
    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)
    return Z, simulate(qstep)

# -------------------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------------------
def run_experiment():
    images = ["lighthouse.mat", "bridge.mat", "flamingo.mat"]
    
    for img_name in images:
        print(f"\nProcessing {img_name}...")
        mat = loadmat(img_name)
        key = next(k for k in mat if not k.startswith('__'))
        X_orig = mat[key].astype(np.float64) - 128.0
        
        # 1. 4x4 LBT (Entropy only, since jpegenc is 8x8)
        print("  Running 4x4 LBT (Entropy)...")
        Z_lbt4_ent, bits_lbt4_ent = lbt_entropy(X_orig, TOTAL_TARGET_BITS, N=4)
        
        # 2. 8x8 LBT (Entropy)
        print("  Running 8x8 LBT (Entropy)...")
        Z_lbt8_ent, bits_lbt8_ent = lbt_entropy(X_orig, TOTAL_TARGET_BITS, N=8)
        
        # 3. 8x8 LBT (Huffman)
        print("  Running 8x8 LBT (Huffman)...")
        Z_lbt8_huff, bits_lbt8_huff = lbt_huffman(X_orig, TOTAL_TARGET_BITS, N=8)
        
        # 4. Reordered 8x8 LBT + Quadtree (Entropy)
        print("  Running Reordered LBT + Quadtree...")
        Yrq, qstep, bits_qt = encode_reordered_lbt_qt(X_orig, TOTAL_TARGET_BITS)
        Z_qt = decode_reordered_lbt_qt(Yrq, qstep)
        
        # --- Plotting ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Compression Strategies Comparison on {img_name}", fontsize=16)
        
        plot_image(Z_lbt4_ent + 128.0, ax=axes[0])
        axes[0].set_title(f"4x4 LBT (Entropy)\nRMS: {rms_error(X_orig, Z_lbt4_ent):.2f} | SSIM: {ssim_score(X_orig, Z_lbt4_ent):.4f}\nBits: {bits_lbt4_ent:.0f}")
        
        plot_image(Z_lbt8_ent + 128.0, ax=axes[1])
        axes[1].set_title(f"8x8 LBT (Entropy)\nRMS: {rms_error(X_orig, Z_lbt8_ent):.2f} | SSIM: {ssim_score(X_orig, Z_lbt8_ent):.4f}\nBits: {bits_lbt8_ent:.0f}")
        
        plot_image(Z_lbt8_huff + 128.0, ax=axes[2])
        axes[2].set_title(f"8x8 LBT (Huffman)\nRMS: {rms_error(X_orig, Z_lbt8_huff):.2f} | SSIM: {ssim_score(X_orig, Z_lbt8_huff):.4f}\nBits: {bits_lbt8_huff:.0f}")
        
        plot_image(Z_qt + 128.0, ax=axes[3])
        axes[3].set_title(f"Reordered 8x8 LBT + Quadtree\nRMS: {rms_error(X_orig, Z_qt):.2f} | SSIM: {ssim_score(X_orig, Z_qt):.4f}\nBits: {bits_qt:.0f}")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()
