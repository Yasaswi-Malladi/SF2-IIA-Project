import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.jpeg import jpegenc, jpegdec

# -------------------------------------------------------------------------
# Mathematical and Coding Theory for Reordered LBT + Quadtree
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

# -------------------------------------------------------------------------
# Quadtree Structure Scanner
# -------------------------------------------------------------------------
def get_qt_structure(Yrq, min_size=4):
    blocks = []
    tree_bits = []
    
    def _split(x, y, size):
        block = Yrq[y:y+size, x:x+size]
        if np.all(block == 0) or size <= min_size:
            tree_bits.append(0)
            blocks.append((x, y, size))
        else:
            tree_bits.append(1)
            half = size // 2
            _split(x, y, half)
            _split(x + half, y, half)
            _split(x, y + half, half)
            _split(x + half, y + half, half)
            
    _split(0, 0, Yrq.shape[0])
    return tree_bits, blocks

# -------------------------------------------------------------------------
# Custom Fixed-Probability (Percent) Deadzone Quantizer
# -------------------------------------------------------------------------
def percent_deadzone_quant(x, T, step):
    xq = np.zeros_like(x)
    mask = np.abs(x) >= T
    # Bins start at T instead of 0
    xq[mask] = np.sign(x[mask]) * (1 + np.floor((np.abs(x[mask]) - T) / step))
    return xq

def percent_deadzone_dequant(xq, T, step):
    x_hat = np.zeros_like(xq, dtype=float)
    mask = np.abs(xq) > 0
    # Reconstruct exact center of the shifted bins
    x_hat[mask] = np.sign(xq[mask]) * (T + (np.abs(xq[mask]) - 1) * step + step / 2)
    return x_hat

def encode_percent_lbt_qt(X, target_bits, centre_quant=0.3):
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    C = dct_ii(8)
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yr = regroup(Y, 8)
    
    def simulate(qstep):
        if centre_quant == 'Uniform':
            T = qstep / 2.0  # Equal step size for every bin
        else:
            T = np.percentile(np.abs(Yr), centre_quant * 100)
            
        Yrq = percent_deadzone_quant(Yr, T, qstep)
        tree_bits, blocks = get_qt_structure(Yrq, min_size=4)
        
        coef_bits = 0
        for (x, y, size) in blocks:
            block = Yrq[y:y+size, x:x+size]
            if not np.all(block == 0):
                coef_bits += bpp(block) * block.size
                
        total_bits = len(tree_bits) + coef_bits
        return total_bits, Yrq, T
        
    lo, hi = 0.1, 500.0  # Raised upper bound just in case
    for _ in range(35):
        mid = (lo + hi) / 2
        bits, _, _ = simulate(mid)
        if bits > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    final_bits, Yrq, final_T = simulate(qstep)
    return Yrq, qstep, final_T, final_bits

def decode_percent_lbt_qt(Yrq, T, qstep):
    Yr_hat = percent_deadzone_dequant(Yrq, T, qstep)
    Y_hat = regroup(Yr_hat, 256//8) 
    
    C = dct_ii(8)
    Xp_hat = colxfm(colxfm(Y_hat.T, C.T).T, C.T)
    
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    X_hat = Xp_hat.copy()
    X_hat[:, t] = colxfm(X_hat[:, t].T, Pr.T).T
    X_hat[t, :] = colxfm(X_hat[t, :], Pr.T)
    
    return X_hat

def perfect_reordered_lbt(X):
    """Encodes and decodes without ANY quantization to test mathematical perfection."""
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
    C = dct_ii(8)
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yr = regroup(Y, 8)
    
    Y_hat = regroup(Yr, 256//8) 
    Xp_hat = colxfm(colxfm(Y_hat.T, C.T).T, C.T)
    X_hat = Xp_hat.copy()
    X_hat[:, t] = colxfm(X_hat[:, t].T, Pr.T).T
    X_hat[t, :] = colxfm(X_hat[t, :], Pr.T)
    
    return Yr, X_hat

# -------------------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------------------
def run_experiment():
    import os
    # Force the working directory to be the project root so it can find the .mat files
    # no matter where the user runs the script from!
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    images = ["lighthouse.mat", "bridge.mat", "flamingo.mat"]
    
    # Mathematical proof showed we need ~88% zeroes to hit 5KB!
    # Sweeping through the critical boundary to see exactly when it succeeds.
    sweep_percentages = ['Uniform', 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
    
    for img_name in images:
        print(f"\n================ Processing {img_name} ================")
        mat = loadmat(img_name)
        key = next(k for k in mat if not k.startswith('__'))
        X_orig = mat[key].astype(np.float64) - 128.0
        
        # ----------------------------------------------------------
        # The Deadzone Parameter Sweep
        # ----------------------------------------------------------
        sweep_results = []
        
        # Add No Quantization baseline
        print("  Running Perfect Reconstruction (No Quantization)...")
        Yr_perf, Z_perf = perfect_reordered_lbt(X_orig)
        sweep_results.append({
            'p': 'No Quant',
            'Yrq': Yr_perf, # Raw floats!
            'qstep': 0,
            'T': 0,
            'Z': Z_perf,
            'bits': float('inf')
        })
        
        for p in sweep_percentages:
            if p == 'Uniform':
                print("  Running Sweep: Uniform (No Deadzone)...")
            else:
                print(f"  Running Sweep: Deadzone = {int(p*100)}% ...")
                
            Yrq_p, qstep_p, T_p, bits_p = encode_percent_lbt_qt(X_orig, TOTAL_TARGET_BITS, centre_quant=p)
            Z_p = decode_percent_lbt_qt(Yrq_p, T_p, qstep_p)
            sweep_results.append({
                'p': p,
                'Yrq': Yrq_p,
                'qstep': qstep_p,
                'T': T_p,
                'Z': Z_p,
                'bits': bits_p
            })
            
        # 1. Plot Histograms
        fig_hist, axes_hist = plt.subplots(1, len(sweep_results), figsize=(28, 4))
        fig_hist.suptitle(f"Quantized Coefficients Histograms ({img_name})", fontsize=16)
        bins = np.arange(-5, 6) - 0.5
        
        for i, res in enumerate(sweep_results):
            if res['p'] == 'No Quant':
                title_str = 'Raw LBT\n(No Quantization)'
                axes_hist[i].hist(res['Yrq'].flatten(), bins=bins, color='gray', alpha=0.7, edgecolor='black')
                axes_hist[i].set_title(title_str)
            else:
                title_str = 'Uniform\n(No Deadzone)' if res['p'] == 'Uniform' else f"Deadzone {int(res['p']*100)}%"
                axes_hist[i].hist(res['Yrq'].flatten(), bins=bins, color='green', alpha=0.7, edgecolor='black')
                axes_hist[i].set_title(f"{title_str}\nQstep={res['qstep']:.2f} | T={res['T']:.2f}")
                
            axes_hist[i].set_xlim([-5, 5])
            axes_hist[i].set_yscale('log')
            
        plt.tight_layout()
        plt.show()
        
        # 2. Plot Reconstructions
        fig_sweep, axes_sweep = plt.subplots(1, len(sweep_results), figsize=(32, 5))
        fig_sweep.suptitle(f"Percent Deadzone Sweep on {img_name} (Target: 5.0 KB)", fontsize=16)
        
        for i, res in enumerate(sweep_results):
            plot_image(res['Z'] + 128.0, ax=axes_sweep[i])
            rms = rms_error(X_orig, res['Z'])
            ssim = ssim_score(X_orig, res['Z'])
            
            if res['p'] == 'No Quant':
                axes_sweep[i].set_title(f"Perfect Reconstruction\n(No Quantization)\nRMS: {rms:.2f} | SSIM: {ssim:.4f}\nBits: N/A")
            else:
                title_str = 'Uniform' if res['p'] == 'Uniform' else f"{int(res['p']*100)}% Deadzone"
                axes_sweep[i].set_title(f"{title_str}\nRMS: {rms:.2f} | SSIM: {ssim:.4f}\nBits: {res['bits']:.0f}")
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()
