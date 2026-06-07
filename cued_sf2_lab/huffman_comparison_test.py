import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Import necessary lab functions
from cued_sf2_lab.dct import dct_ii, colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import rowdec, rowint, bpp
from cued_sf2_lab.jpeg import jpegenc, jpegdec, huffdflt, huffgen, huffenc, runampl, diagscan
from cued_sf2_lab.familiarisation import plot_image, load_mat_img

TARGET_BITS = 40960  # 5.0 KB
MIN_BLOCK = 8
MAX_BLOCK = 32
MAX_DEPTH = int(np.log2(MAX_BLOCK // MIN_BLOCK))
DCBITS = 10
DEPTH_SCALE = 1.3

# --- Helper Functions ---
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

def block_edge_density(block):
    if block.shape[0] < 2 or block.shape[1] < 2: return 0.0
    gx = np.abs(np.diff(block, axis=1))
    gy = np.abs(np.diff(block, axis=0))
    edge_thresh = 5.0
    num_edges = np.sum(gx > edge_thresh) + np.sum(gy > edge_thresh)
    return num_edges / (gx.size + gy.size)

def deadzone_quant(x, step):
    return np.trunc(x / step)

def deadzone_dequant(xq, step):
    return np.where(xq == 0, 0, np.sign(xq) * (np.abs(xq) + 0.5) * step)

def advanced_smoothing(image, qstep, strength=1.0):
    blurred = nd.gaussian_filter(image, sigma=1.2 * strength)
    diff = np.abs(image - blurred)
    noise_thresh = qstep * 0.8 * strength
    weight = np.clip(diff / noise_thresh, 0.0, 1.0)
    return image * weight + blurred * (1.0 - weight)

def block_adaptive_smoothing(image, qstep, block_size=4, strength=1.0, passes=1):
    H, W = image.shape
    smoothed_full = advanced_smoothing(image, qstep, strength)
    final_image = np.copy(image)
    noise_var = (qstep ** 2) / 12.0
    h_blocks = H // block_size
    w_blocks = W // block_size
    weight_map = np.zeros((h_blocks, w_blocks))
    
    for y_idx in range(h_blocks):
        for x_idx in range(w_blocks):
            y = y_idx * block_size
            x = x_idx * block_size
            block = image[y:y+block_size, x:x+block_size]
            b_var = np.var(block)
            weight_map[y_idx, x_idx] = np.clip(b_var / (noise_var * 2.0 * strength), 0.0, 1.0)
            
    for _ in range(passes - 1):
        weight_map = nd.uniform_filter(weight_map, size=3, mode='nearest')
        
    for y_idx in range(h_blocks):
        for x_idx in range(w_blocks):
            y = y_idx * block_size
            x = x_idx * block_size
            block = image[y:y+block_size, x:x+block_size]
            w = weight_map[y_idx, x_idx]
            final_image[y:y+block_size, x:x+block_size] = (
                block * w + smoothed_full[y:y+block_size, x:x+block_size] * (1.0 - w)
            )
    return final_image

# --- Quadtree Common Logic ---
def get_qt_structure(image, var_thresh):
    H, W = image.shape
    tree_bits = []
    blocks = [] # List of (x, y, size, depth)
    def _split(x, y, size, depth):
        block = image[y:y+size, x:x+size]
        if size > MIN_BLOCK and depth < MAX_DEPTH and block_edge_density(block) > var_thresh:
            tree_bits.append(1)
            half = size // 2
            for dx, dy in [(0, 0), (half, 0), (0, half), (half, half)]:
                _split(x + dx, y + dy, half, depth + 1)
        else:
            tree_bits.append(0)
            blocks.append((x, y, size, depth))
    for y in range(0, H, MAX_BLOCK):
        for x in range(0, W, MAX_BLOCK):
            _split(x, y, MAX_BLOCK, 0)
    return tree_bits, blocks

def find_qt_thresh(X):
    # Fixed absolute threshold of 5% edge pixels to avoid the "percentile trap"
    return 0.05 

# --- ENTROPY ENCODERS ---
def qt_entropy(X, target_bits):
    var_thresh = find_qt_thresh(X)
    tree_bits, blocks = get_qt_structure(X, var_thresh)
    
    def simulate(qstep):
        Yq_all = []
        for (x, y, size, depth) in blocks:
            block = X[y:y+size, x:x+size]
            C = dct_ii(size)
            Y = colxfm(colxfm(block, C).T, C).T
            q = qstep * (DEPTH_SCALE ** depth)
            Yq = deadzone_quant(Y, q).astype(int)
            Yq_all.append(Yq.flatten())
        
        all_coeffs = np.concatenate(Yq_all)
        entropy_bits = bpp(all_coeffs) * all_coeffs.size
        return entropy_bits + len(tree_bits)

    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits:
            lo = mid
        else:
            hi = mid
    
    qstep = (lo + hi) / 2
    Z = np.zeros_like(X)
    for (x, y, size, depth) in blocks:
        block = X[y:y+size, x:x+size]
        C = dct_ii(size)
        Y = colxfm(colxfm(block, C).T, C).T
        q = qstep * (DEPTH_SCALE ** depth)
        Yq = deadzone_quant(Y, q).astype(int)
        Zi = deadzone_dequant(Yq, q)
        Z[y:y+size, x:x+size] = colxfm(colxfm(Zi.T, C.T).T, C.T)
        
    return Z, simulate(qstep), qstep

def qt_entropy_smoothed(X, target_bits):
    Z, bits, qstep = qt_entropy(X, target_bits)
    Z_smooth = advanced_smoothing(Z, qstep, strength=1.0)
    return Z_smooth, bits, qstep

def qt_entropy_adaptive(X, target_bits):
    Z, bits, qstep = qt_entropy(X, target_bits)
    Z_adapt = block_adaptive_smoothing(Z, qstep, block_size=4, strength=1.0, passes=1)
    return Z_adapt, bits, qstep

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
    return Z, simulate(qstep), qstep

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
    return Z, simulate(qstep), qstep

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
    return Z, simulate(qstep), qstep

def lap_entropy(X, target_bits):
    n_levels = 3
    h = np.array([1, 2, 1]) / 4.0

    layers = []
    current = X.copy()
    for i in range(n_levels):
        decimated = rowdec(rowdec(current, h).T, h).T
        interpolated = rowint(rowint(decimated, 2*h).T, 2*h).T
        diff = current - interpolated
        layers.append(diff)
        current = decimated
    layers.append(current)
    
    def simulate(qstep):
        bits = 0
        for i, layer in enumerate(layers):
            q = qstep if i < len(layers)-1 else qstep/2
            Yq = deadzone_quant(layer, q)
            bits += bpp(Yq) * Yq.size
        return bits
        
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    q_layers = []
    for i, layer in enumerate(layers):
        q = qstep if i < len(layers)-1 else qstep/2
        q_layers.append(deadzone_dequant(deadzone_quant(layer, q), q))
        
    recon = q_layers[-1]
    for i in range(n_levels - 1, -1, -1):
        recon = rowint(rowint(recon, 2*h).T, 2*h).T + q_layers[i]
    return recon, simulate(qstep), qstep

# --- HUFFMAN ENCODERS ---
def qt_huffman(X, target_bits):
    var_thresh = find_qt_thresh(X)
    tree_bits, blocks = get_qt_structure(X, var_thresh)
    
    huffhist = np.zeros(16**2)
    dhufftab = huffdflt(1)
    _, ehuf = huffgen(dhufftab)
    
    def simulate(qstep):
        vlc_list = []
        for (x, y, size, depth) in blocks:
            block = X[y:y+size, x:x+size]
            C = dct_ii(size)
            Y = colxfm(colxfm(block, C).T, C).T
            q = qstep * (DEPTH_SCALE ** depth)
            Yq = deadzone_quant(Y, q).astype(int)
            
            yqflat = Yq.flatten('F')
            dccoef = int(yqflat[0]) + 2**(DCBITS-1)
            dccoef = np.clip(dccoef, 0, 2**DCBITS - 1)
            vlc_list.append(np.array([[dccoef, DCBITS]]))
            
            scan = diagscan(size)
            ra = runampl(yqflat[scan])
            if ra.shape[0] > 0:
                vlc_list.append(huffenc(huffhist, ra, ehuf))
            
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc_list)
        return int(np.sum(vlc[:, 1])) + len(tree_bits)

    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    
    Z = np.zeros_like(X)
    for (x, y, size, depth) in blocks:
        block = X[y:y+size, x:x+size]
        C = dct_ii(size)
        Y = colxfm(colxfm(block, C).T, C).T
        q = qstep * (DEPTH_SCALE ** depth)
        Yq = deadzone_quant(Y, q).astype(int)
        Zi = deadzone_dequant(Yq, q)
        Z[y:y+size, x:x+size] = colxfm(colxfm(Zi.T, C.T).T, C.T)
        
    return Z, simulate(qstep), qstep

def dct_huffman(X, target_bits):
    def simulate(qstep):
        vlc, _ = jpegenc(X, qstep, log=False)
        return int(np.sum(vlc[:, 1]))
        
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if simulate(mid) > target_bits: lo = mid
        else: hi = mid
        
    qstep = (lo + hi) / 2
    vlc, _ = jpegenc(X, qstep, log=False)
    Z = jpegdec(vlc, qstep, log=False)
    return Z, int(np.sum(vlc[:, 1])), qstep

def lbt_huffman(X, target_bits):
    s = (1 + 5**0.5) / 2
    Pf, Pr = pot_ii(8, s)
    t = np.s_[4:-4]
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    
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
    return Z, int(np.sum(vlc[:, 1])), qstep

# --- MAIN RUNNER ---
def main():
    images = ["lighthouse.mat", "bridge.mat", "flamingo.mat"]
    
    for img_name in images:
        mat = loadmat(img_name)
        key = next(k for k in mat if not k.startswith('__'))
        X = mat[key]
        
        X = X - 128.0
        
        print(f"\n=====================================")
        print(f"PROCESSING {img_name} @ {TARGET_BITS/8/1024:.1f} KB")
        print(f"=====================================")
        
        # 1. ENTROPY PLOT
        fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4))
        fig1.suptitle(f"{img_name} - Theoretical Entropy (No Huffman Headers) - {TARGET_BITS/8/1024:.1f} KB", fontsize=14, fontweight='bold')
        
        methods_entropy = [
            ("Quadtree Base", qt_entropy),
            ("Quadtree Smooth", qt_entropy_smoothed),
            ("Quadtree Adapt", qt_entropy_adaptive),
            ("DCT 8x8", dct_entropy),
            ("LBT 8x8", lbt_entropy)
        ]
        
        print("\n--- Entropy Comparison ---")
        for idx, (name, func) in enumerate(methods_entropy):
            Z, bits, q = func(X, TARGET_BITS)
            ssim = ssim_score(X, Z)
            print(f"  {name:<15}: Bits={bits:.0f}, SSIM={ssim:.4f}, Q={q:.2f}")
            plot_image(Z + 128.0, ax=axes1[idx])
            axes1[idx].set_title(f"{name}\nSSIM: {ssim:.4f}")
            axes1[idx].axis('off')
            
        plt.tight_layout()
        plt.show(block=False)
        
        # 2. HUFFMAN PLOT
        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        fig2.suptitle(f"{img_name} - Actual Huffman Encoding (Bitstream + Headers) - {TARGET_BITS/8/1024:.1f} KB", fontsize=14, fontweight='bold')
        
        methods_huffman = [
            ("Quadtree", qt_huffman),
            ("DCT 8x8", dct_huffman),
            ("LBT 8x8", lbt_huffman)
        ]
        
        print("\n--- Huffman Comparison ---")
        for idx, (name, func) in enumerate(methods_huffman):
            Z, bits, q = func(X, TARGET_BITS)
            ssim = ssim_score(X, Z)
            print(f"  {name:<15}: Bits={bits:.0f}, SSIM={ssim:.4f}, Q={q:.2f}")
            plot_image(Z + 128.0, ax=axes2[idx])
            axes2[idx].set_title(f"{name}\nSSIM: {ssim:.4f}")
            axes2[idx].axis('off')
            
        plt.tight_layout()
        plt.show(block=True) # Block to show plots sequentially

if __name__ == "__main__":
    main()
