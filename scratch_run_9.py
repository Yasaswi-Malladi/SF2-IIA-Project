#%matplotlib widget
import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple

X, _ = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
X = X - 128.0
h1 = np.array([-1, 2, 6, 2, -1])/8
h2 = np.array([-1, 2, -1])/4

from cued_sf2_lab.laplacian_pyramid import rowdec
U = rowdec(X, h1)

from cued_sf2_lab.laplacian_pyramid import rowdec2
V = rowdec2(X, h2)

# Display U and V and print their standard deviations
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_image(U, ax=axes[0])
axes[0].set_title(f"U (Low-pass) Std={np.std(U):.2f}")

plot_image(V, ax=axes[1])
axes[1].set_title(f"V (High-pass) Std={np.std(V):.2f}")
plt.tight_layout()


UU = rowdec(U.T, h1).T
UV = rowdec2(U.T, h2).T
VU = rowdec(V.T, h1).T
VV = rowdec2(V.T, h2).T

# Display the block matrix, multiplying high-pass components by 5 for visualization
k = 5
dwt_vis = np.block([
    [UU, VU * k],
    [UV * k, VV * k]
])

fig, ax = plt.subplots(figsize=(8, 8))
plot_image(dwt_vis, ax=ax)
ax.set_title("1-Level DWT (High-pass components scaled by 5)")


from cued_sf2_lab.laplacian_pyramid import rowint, rowint2

g1 = np.array([1, 2, 1])/2
g2 = np.array([-1, -2, 6, -2, -1])/4
Ur = rowint(UU.T, g1).T + rowint2(UV.T, g2).T
Vr = rowint(VU.T, g1).T + rowint2(VV.T, g2).T

# Verify that Ur and Vr reconstruct U and V perfectly
print("Difference between Ur and U:", np.max(np.abs(Ur - U)))
print("Difference between Vr and V:", np.max(np.abs(Vr - V)))


# demonstrator answer here
np.testing.assert_equal(Ur, U)
np.testing.assert_equal(Vr, V)

Xr = rowint(Ur,g1) + rowint2(Vr,g2)

# Verify that Xr reconstructs X perfectly
print("Difference between reconstructed Xr and original X:", np.max(np.abs(Xr - X)))


from cued_sf2_lab.dwt import dwt
IPython.display.Code(inspect.getsource(dwt), language="python")

from cued_sf2_lab.dwt import idwt
IPython.display.Code(inspect.getsource(idwt), language="python")

Y = dwt(X)
Xr = idwt(Y)

fig, axs = plt.subplots(1, 2)
plot_image(Y, ax=axs[0])
axs[0].set(title="Y")
plot_image(Xr, ax=axs[1])
axs[1].set(title="Xr");

# Implement a 4-level DWT by iteratively applying dwt to the top-left sub-image
from cued_sf2_lab.dwt import dwt, idwt

m = 256
Y = X.copy()

for level in range(4):
    Y[:m, :m] = dwt(Y[:m, :m])
    m = m // 2

# Visualise the multilevel DWT
fig, ax = plt.subplots(figsize=(8, 8))
plot_image(Y, ax=ax)
ax.set_title("4-Level DWT (Quaternary Tree)")


# Reconstruct the image from the 4-level DWT coefficients
m = 32
Xr = Y.copy()

for level in range(4):
    Xr[:m, :m] = idwt(Xr[:m, :m])
    m = m * 2

max_err = np.max(np.abs(X - Xr))
print("4-Level DWT Reversibility Verification:")
print(f"  Maximum absolute error between X and reconstructed Xr: {max_err:.2e}")


def nlevdwt(X, n):
    """
    Perform n-level 2D Discrete Wavelet Transform.
    """
    Y = X.copy()
    m = Y.shape[0]
    for level in range(n):
        Y[:m, :m] = dwt(Y[:m, :m])
        m = m // 2
    return Y

def nlevidwt(Y, n):
    """
    Perform n-level 2D Inverse Discrete Wavelet Transform.
    """
    Xr = Y.copy()
    m = Xr.shape[0] // (2**(n-1))
    for level in range(n):
        Xr[:m, :m] = idwt(Xr[:m, :m])
        m = m * 2
    return Xr


# Test nlevdwt and nlevidwt for perfect reconstruction
n_levels = 4
Y_test = nlevdwt(X, n_levels)
Xr_test = nlevidwt(Y_test, n_levels)

max_err_test = np.max(np.abs(X - Xr_test))
print(f"nlevdwt/nlevidwt ({n_levels} levels) test:")
print(f"  Maximum absolute error: {max_err_test:.2e}")


from cued_sf2_lab.laplacian_pyramid import quantise, bpp

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes DWT coefficients Y and calculates their entropies.
    """
    n = dwtstep.shape[1] - 1  # number of levels
    Yq = Y.copy()
    dwtent = np.zeros_like(dwtstep)
    
    height, width = Y.shape
    
    # Process each level i from 0 to n-1
    for i in range(n):
        m_h = height // (2**i)
        m_w = width // (2**i)
        
        # Extract high-pass sub-images (TR, BL, BR)
        tr = Yq[:m_h//2, m_w//2:m_w]
        bl = Yq[m_h//2:m_h, :m_w//2]
        br = Yq[m_h//2:m_h, m_w//2:m_w]
        
        # Quantise and calculate entropy
        # Top-Right (k=0)
        tr_q = quantise(tr, dwtstep[0, i])
        Yq[:m_h//2, m_w//2:m_w] = tr_q
        dwtent[0, i] = bpp(tr_q)
        
        # Bottom-Left (k=1)
        bl_q = quantise(bl, dwtstep[1, i])
        Yq[m_h//2:m_h, :m_w//2] = bl_q
        dwtent[1, i] = bpp(bl_q)
        
        # Bottom-Right (k=2)
        br_q = quantise(br, dwtstep[2, i])
        Yq[m_h//2:m_h, m_w//2:m_w] = br_q
        dwtent[2, i] = bpp(br_q)
        
    # Process final low-pass image (top-left of the smallest level, i = n-1)
    m_h = height // (2**(n-1))
    m_w = width // (2**(n-1))
    lp = Yq[:m_h//2, :m_w//2]
    
    lp_q = quantise(lp, dwtstep[0, n])
    Yq[:m_h//2, :m_w//2] = lp_q
    dwtent[0, n] = bpp(lp_q)
    
    return Yq, dwtent


# Test quantdwt with equal step size of 17 on a 3-level DWT
def get_dwt_bpp(dwtent):
    n = dwtent.shape[1] - 1
    total_bpp = 0.0
    for i in range(n):
        weight = 1.0 / (4 ** (i + 1))
        total_bpp += (dwtent[0, i] + dwtent[1, i] + dwtent[2, i]) * weight
    total_bpp += dwtent[0, n] * (1.0 / (4 ** n))
    return total_bpp

n_levels = 3
dwtstep = np.ones((3, n_levels + 1)) * 17.0

Y = nlevdwt(X, n_levels)
Yq, dwtent = quantdwt(Y, dwtstep)
Z = nlevidwt(Yq, n_levels)

std_err = np.std(X - Z)
bits = get_dwt_bpp(dwtent)

print(f"Equal-step-size DWT ({n_levels} levels, step=17):")
print(f"  Reconstructed RMS Error: {std_err:.4f}")
print(f"  Weighted Bit Rate: {bits:.4f} bits/pixel")


Xb, _ = load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})
Xb = Xb - 128.0

fig, ax = plt.subplots()
plot_image(Xb, ax=ax)
ax.set(title="bridge.mat");

from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Equal-MSE step-size scaling matrix generator based on synthesis filter noise gains
def get_dwt_equal_mse_steps(base_step, n):
    g1 = np.array([1, 2, 1])/2
    g2 = np.array([-1, -2, 6, -2, -1])/4
    g1_gain = np.sum(g1**2) # 1.5
    g2_gain = np.sum(g2**2) # 2.875
    
    dwtstep = np.zeros((3, n + 1))
    for i in range(n):
        lp_cum_gain = (g1_gain**2)**i
        gain_tr_bl = lp_cum_gain * (g1_gain * g2_gain)
        dwtstep[0, i] = base_step / np.sqrt(gain_tr_bl)
        dwtstep[1, i] = base_step / np.sqrt(gain_tr_bl)
        gain_br = lp_cum_gain * (g2_gain**2)
        dwtstep[2, i] = base_step / np.sqrt(gain_br)
        
    gain_lp = (g1_gain**2)**n
    dwtstep[0, n] = base_step / np.sqrt(gain_lp)
    return dwtstep

# Evaluate both images: Lighthouse and Bridge
for img_name, img_data in [('Lighthouse', X), ('Bridge', Xb)]:
    # Get target RMS error from direct quantization of image with step 17
    ref_q = quantise(img_data, 17)
    target_std = np.std(img_data - ref_q)
    ref_bits = bpp(ref_q)
    
    print(f"\n==================== IMAGE: {img_name} (Target RMS Error = {target_std:.2f}, Direct Bits = {ref_bits:.2f}) ====================")
    print("{:<8} | {:<10} | {:<15} | {:<12} | {:<12}".format("Levels n", "Scheme", "Base Step", "Bit Rate", "Comp. Ratio"))
    print("-" * 65)
    
    # Create a 2x3 figure for this image's reconstructions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for col_idx, n in enumerate([3, 4, 5]):
        Y_img = nlevdwt(img_data, n)
        
        # 1. Equal-step size scheme
        def check_equal_step(step):
            steps = np.ones((3, n + 1)) * step
            Yq, _ = quantdwt(Y_img, steps)
            Z = nlevidwt(Yq, n)
            return (np.std(img_data - Z) - target_std) ** 2
            
        res_eq = minimize_scalar(check_equal_step, bounds=(5, 40), method='bounded')
        opt_step_eq = res_eq.x
        
        steps_eq = np.ones((3, n + 1)) * opt_step_eq
        Yq_eq, dwtent_eq = quantdwt(Y_img, steps_eq)
        bits_eq = get_dwt_bpp(dwtent_eq)
        ratio_eq = ref_bits / bits_eq
        
        # Reconstruct and plot Equal-Step image
        Z_eq = nlevidwt(Yq_eq, n)
        plot_image(Z_eq, ax=axes[0, col_idx])
        axes[0, col_idx].set_title(f"Equal-Step (n={n})\nStep: {opt_step_eq:.1f}, Bits: {bits_eq:.2f}")

        print("{:<8} | {:<10} | {:<15.2f} | {:<12.3f} | {:<12.2f}".format(n, "Equal-Step", opt_step_eq, bits_eq, ratio_eq))
        
        # 2. Equal-MSE scheme
        def check_equal_mse(base_step):
            steps = get_dwt_equal_mse_steps(base_step, n)
            Yq, _ = quantdwt(Y_img, steps)
            Z = nlevidwt(Yq, n)
            return (np.std(img_data - Z) - target_std) ** 2
            
        res_mse = minimize_scalar(check_equal_mse, bounds=(5, 100), method='bounded')
        opt_step_mse = res_mse.x
        
        steps_mse = get_dwt_equal_mse_steps(opt_step_mse, n)
        Yq_mse, dwtent_mse = quantdwt(Y_img, steps_mse)
        bits_mse = get_dwt_bpp(dwtent_mse)
        ratio_mse = ref_bits / bits_mse
        
        # Reconstruct and plot Equal-MSE image
        Z_mse = nlevidwt(Yq_mse, n)
        plot_image(Z_mse, ax=axes[1, col_idx])
        axes[1, col_idx].set_title(f"Equal-MSE (n={n})\nBase: {opt_step_mse:.1f}, Bits: {bits_mse:.2f}")

        print("{:<8} | {:<10} | {:<15.2f} | {:<12.3f} | {:<12.2f}".format(n, "Equal-MSE", opt_step_mse, bits_mse, ratio_mse))
        print("-" * 65)
        
    fig.suptitle(f"Reconstructed Images for {img_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()


