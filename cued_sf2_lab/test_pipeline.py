import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import plot_image
import os

from encode import encode, header_bits
from decode import decode

def test_pipeline():
    # Load lighthouse.mat from the current directory or parent directory
    try:
        mat = loadmat("lighthouse.mat")
    except FileNotFoundError:
        mat = loadmat("../lighthouse.mat")
        
    key = next(k for k in mat if not k.startswith('__'))
    X_orig = mat[key].astype(np.float64)
    
    print("Encoding image...")
    vlc, header = encode(X_orig)
    
    bits = header_bits(header)
    print(f"Target was: 40000 bits")
    print(f"Actual compressed size: {bits} bits")
    
    print("Decoding image...")
    X_hat = decode(vlc, header)
    
    rms = np.std(X_orig - X_hat)
    print(f"RMS Error: {rms:.2f}")
    
    print("Pipeline executed successfully!")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_image(X_orig, ax=axes[0])
    axes[0].set_title("Original Image")
    
    plot_image(X_hat, ax=axes[1])
    axes[1].set_title(f"Decoded Image (RMS: {rms:.2f})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_pipeline()
