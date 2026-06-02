import matplotlib.pyplot as plt

from cued_sf2_lab.dct import dct_ii

C8 = dct_ii(8)

import inspect
import IPython.display
IPython.display.Code(inspect.getsource(dct_ii), language="python")

fig, ax = plt.subplots()
ax.plot(C8.T);

import numpy as np
import matplotlib.pyplot as plt

N = 8
# Create a high-resolution grid for the continuous smooth curves
t = np.linspace(-0.5, N - 0.5, 500)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each basis function
for k in range(N):
    # Calculate the continuous curve
    if k == 0:
        y_smooth = np.ones_like(t) / np.sqrt(N)
    else:
        y_smooth = np.sqrt(2/N) * np.cos(k * (t + 0.5) * np.pi / N)
    
    # Get the 8 discrete points
    n_discrete = np.arange(N)
    if k == 0:
        y_discrete = np.ones_like(n_discrete) / np.sqrt(N)
    else:
        y_discrete = np.sqrt(2/N) * np.cos(k * (n_discrete + 0.5) * np.pi / N)
        
    # Plot the smooth line and discrete points
    line = ax.plot(t, y_smooth, label=f'Row {k} (k={k})', alpha=0.8)
    color = line[0].get_color()
    ax.scatter(n_discrete, y_discrete, color=color, s=40, zorder=3)

ax.set_title('DCT-II Basis Functions: Smooth Curves and Discrete Samples', fontsize=14, fontweight='bold')
ax.set_xlabel('Sample index (n)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_xlim(-0.7, N - 0.3)
ax.set_xticks(range(N))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="DCT Frequencies")
plt.tight_layout()


from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.dct import colxfm

X_pre_zero_mean, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
X = X_pre_zero_mean - 128.0

Y = colxfm(colxfm(X, C8).T, C8).T

from cued_sf2_lab.familiarisation import plot_image

fig, ax = plt.subplots()
plot_image(Y, ax=ax);

from cued_sf2_lab.dct import regroup

N = 8
fig, ax = plt.subplots()
plot_image(regroup(Y, N)/N, ax=ax);



# Your code here
# energy calculations
def calculate_dct_energies(Y, N=8):
    Yr = regroup(Y,N)/N   # scaled by 64 down coefficients
    # 2. Get the dimensions of the image and the size of each sub-image (e.g., 32x32)
    height, width = Y.shape
    sub_height = height // N  # 256 / 8 = 32
    sub_width = width // N    # 256 / 8 = 32
    
    # 3. Initialize an 8x8 matrix to store the energies
    energies = np.zeros((N, N))
    
    # 4. Loop through each of the 8x8 frequency slots
    for u in range(N):
        for v in range(N):
            # Extract the 32x32 sub-image for frequency (u, v)
            sub_image = Yr[u * sub_height : (u + 1) * sub_height, 
                           v * sub_width : (v + 1) * sub_width]
            
            # Calculate energy (mean of the squared values)
            energies[u, v] = np.mean(sub_image ** 2)
            
    return energies
import pandas as pd

# Calculate the energies
energies = calculate_dct_energies(Y, 8)

# Convert to a DataFrame and format to 1 decimal place (suppresses scientific notation)
df_energies = pd.DataFrame(energies)
df_energies.style.format("{:.1f}")


Z = colxfm(colxfm(Y.T, C8.T).T, C8.T)

fig, ax = plt.subplots()
plot_image(Z, ax=ax);

# Your code here
# Calculate the maximum absolute error
max_error = np.max(np.abs(X - Z))
print("Maximum absolute error:", max_error)


import numpy as np
# Stack some NaNs
bases = np.concatenate([np.full((8, 1), np.nan), C8, np.full((8, 1), np.nan)], axis=1)
# Reshape
bases_flat = np.reshape(bases, (-1, 1))

fig, ax = plt.subplots()
im = plot_image(255*bases_flat@bases_flat.T, ax=ax)
fig.colorbar(im);

from cued_sf2_lab.laplacian_pyramid import bpp

def dctbpp(Yr, N):
    """
    Calculates the total entropy of the regrouped DCT coefficients.
    
    Theory:
    In a transform coder, we partition the image into N x N blocks and apply the DCT. 
    By regrouping the coefficients, we obtain N x N sub-images, each containing a single 
    frequency component across all blocks. Assuming these sub-images are coded independently, 
    the total bit rate is the sum of the bits required for each sub-image. 
    We compute the entropy (bpp) of each sub-image, multiply by its size (pixels), sum them 
    up, and divide by the total image size to get the average bits per pixel.
    """
    m, n = Yr.shape
    sub_m = m // N
    sub_n = n // N
    total_bits = 0.0
    for u in range(N):
        for v in range(N):
            sub_image = Yr[u*sub_m : (u+1)*sub_m, v*sub_n : (v+1)*sub_n]
            # Multiply entropy per pixel by the number of pixels in this sub-image
            total_bits += bpp(sub_image) * sub_image.size
    return total_bits / Yr.size


# Quantise Y with a step size of 17
from cued_sf2_lab.laplacian_pyramid import quantise

step = 17
Yq = quantise(Y, step)
Yqr = regroup(Yq, 8)

# Visualise the regrouped coefficients
fig, ax = plt.subplots(figsize=(6, 6))
plot_image(Yqr / 8, ax=ax)
ax.set_title("Regrouped Quantised DCT Coefficients (Step = 17)")

# Calculate bits using independent coding (dctbpp) vs joint coding (bpp)
bits_independent = dctbpp(Yqr, 8)
# bpp(Yq) calculates entropy of the entire image jointly
bits_joint = bpp(Yq)

print(f"Entropy with independent frequency coding (dctbpp): {bits_independent:.4f} bits/pixel")
print(f"Entropy with joint coding (bpp): {bits_joint:.4f} bits/pixel")
print(f"Difference: {bits_joint - bits_independent:.4f} bits/pixel")


# Reconstruct the image Z from the quantised coefficients Yq using inverse DCT-III
# Z = C8^T @ Yq @ C8
Z = colxfm(colxfm(Yq, C8.T).T, C8.T).T

# Calculate RMS error (standard deviation of the difference)
std_dct = np.std(X - Z)

# Direct quantisation of the original image X with a step size of 17
Xq = quantise(X, 17)
std_direct = np.std(X - Xq)

print(f"RMS Error for DCT Quantisation (Step=17): {std_dct:.4f}")
print(f"RMS Error for Direct Quantisation (Step=17): {std_direct:.4f}")


from scipy.optimize import minimize_scalar

# Target RMS error is the error from direct quantisation of X
target_std = std_direct

def get_dct_rms_diff(step_size):
    # Quantise DCT coefficients with trial step size
    Yq_trial = quantise(Y, step_size)
    # Reconstruct
    Z_trial = colxfm(colxfm(Yq_trial, C8.T).T, C8.T).T
    # Calculate RMS error
    std_trial = np.std(X - Z_trial)
    # Return squared difference from target
    return (std_trial - target_std) ** 2

# Find optimal step size using scipy bounded search
res = minimize_scalar(get_dct_rms_diff, bounds=(5, 30), method='bounded')
optimal_step = res.x

print(f"Optimal step size for DCT: {optimal_step:.4f}")

# Verify actual RMS error matches the target
Yq_opt = quantise(Y, optimal_step)
Z_opt = colxfm(colxfm(Yq_opt, C8.T).T, C8.T).T
print(f"Target RMS Error: {target_std:.4f}")
print(f"Actual RMS Error with optimal DCT step: {np.std(X - Z_opt):.4f}")


# Calculate entropy (bits per pixel)
bits_direct = bpp(Xq)
Yq_opt_regrouped = regroup(Yq_opt, 8)
bits_dct = dctbpp(Yq_opt_regrouped, 8)

print(f"Direct Quantisation Bit Rate: {bits_direct:.4f} bits/pixel")
print(f"DCT Quantisation Bit Rate: {bits_dct:.4f} bits/pixel")
print(f"Compression Ratio (Direct / DCT): {bits_direct / bits_dct:.2f}")
print(f"Compression Ratio relative to uncompressed 8-bit: {8.0 / bits_dct:.2f}")

# Plot a visual comparison of the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_image(X, ax=axes[0])
axes[0].set_title("Original Zero-Mean Image (X)")

plot_image(Xq, ax=axes[1])
axes[1].set_title(f"Direct Quantised (Step=17)\nBits: {bits_direct:.2f}")

plot_image(Z_opt, ax=axes[2])
axes[2].set_title(f"DCT Reconstructed (Step={optimal_step:.2f})\nBits: {bits_dct:.2f}")

plt.tight_layout()


# Generate 4-point and 16-point 1-D Type-II DCT matrices using the dct_ii function
C4 = dct_ii(4)
C16 = dct_ii(16)

print("C4 Matrix:")
print(C4)
print("\nC16 Matrix Shape:", C16.shape)


# Let's perform step optimization, compression ratio calculation, and plotting for N = 4, 8, 16
target_std = std_direct
results = {}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (N_val, C_mat) in enumerate([(4, C4), (8, C8), (16, C16)]):
    # 1. Forward 2D DCT transform using N-point transform matrix C_mat
    Y_val = colxfm(colxfm(X, C_mat).T, C_mat).T
    
    # 2. Define step size optimization function
    def check_step(step):
        Yq_val = quantise(Y_val, step)
        Z_val = colxfm(colxfm(Yq_val, C_mat.T).T, C_mat.T).T
        std_val = np.std(X - Z_val)
        return (std_val - target_std) ** 2
        
    res = minimize_scalar(check_step, bounds=(5, 40), method='bounded')
    opt_step = res.x
    
    # 3. Calculate actual bit rate with optimal step size
    Yq_opt_val = quantise(Y_val, opt_step)
    Yr_opt_val = regroup(Yq_opt_val, N_val)
    bits_val = dctbpp(Yr_opt_val, N_val)
    
    # 4. Reconstruct the image for visualization
    Z_val = colxfm(colxfm(Yq_opt_val, C_mat.T).T, C_mat.T).T
    
    # Store results for final table comparison
    results[N_val] = {
        'opt_step': opt_step,
        'bits': bits_val,
        'ratio': bits_direct / bits_val
    }
    
    # Plot reconstructed image
    plot_image(Z_val, ax=axes[idx])
    axes[idx].set_title(f"DCT {N_val}x{N_val}\nStep: {opt_step:.2f}, Bits: {bits_val:.2f}")
    
    print(f"--- {N_val}x{N_val} DCT ---")
    print(f"  Optimal Step Size: {opt_step:.4f}")
    print(f"  Bit Rate: {bits_val:.4f} bits/pixel")
    print(f"  Compression Ratio (Direct / DCT): {bits_direct / bits_val:.2f}")

plt.tight_layout()


# Calculate bits when N = 256 (each pixel treated as its own frequency sub-image)
# Since the image is 256x256, regroup(Y, 256) will yield 256x256 sub-images of size 1x1.
Yqr_8 = regroup(Yq_opt, 8)
bits_256 = dctbpp(Yqr_8, 256)
print(f"Entropy calculated with independent pixels (N=256): {bits_256:.4f} bits/pixel")


# Display final comparison table
print("DCT Transform Size Comparison (Target RMS Error = {:.2f}):".format(target_std))
print("-" * 75)
print("{:<15} | {:<20} | {:<20} | {:<15}".format("Size (NxN)", "Optimal Step Size", "Entropy (bits/pixel)", "Compression Ratio"))
print("-" * 75)
for N_val in [4, 8, 16]:
    r = results[N_val]
    print("{:<15} | {:<20.2f} | {:<20.3f} | {:<15.2f}".format(
        f"{N_val}x{N_val}", r['opt_step'], r['bits'], r['ratio']
    ))
print("-" * 75)


