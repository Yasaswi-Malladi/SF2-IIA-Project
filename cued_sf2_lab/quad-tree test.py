from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.familiarisation import load_mat_img
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from cued_sf2_lab.jpeg import jpegenc, jpegdec

# --- ESSENTIAL HANDOUT INTERFACE ---
# Ensure jpegenc/jpegdec are available from your project's utils
# from sf2_utils import jpegenc, jpegdec

def apply_dct2(block): return dctn(block, type=2, norm='ortho')
def apply_idct2(block): return idctn(block, type=2, norm='ortho')

# ==========================================
# 1. ADAPTIVE FILTERING & QUADTREE LOGIC
# ==========================================
import numpy as np
import scipy.ndimage as nd

def advanced_smoothing(image, qstep, strength=1.0):
    """
    Applies an edge-preserving smoothing filter without relying on skimage.
    We heavily blur the image, then blend it back with the original based on
    how much the pixel changed. If the change is less than the expected
    quantization noise, we keep the smooth version. If the change is large,
    it's a real edge, so we keep the sharp original version.
    """
    # Create a completely smooth version of the image
    blurred = nd.gaussian_filter(image, sigma=1.2 * strength)
    
    # Calculate the absolute difference (sharpness)
    diff = np.abs(image - blurred)
    
    # Threshold for what we consider "quantization block noise" vs a "real edge"
    noise_thresh = qstep * 0.8 * strength
    
    # Weight: 1.0 = use original (sharp edge), 0.0 = use blurred (flat block noise)
    weight = np.clip(diff / noise_thresh, 0.0, 1.0)
    
    # Soft blend
    return image * weight + blurred * (1.0 - weight)

def block_adaptive_smoothing(image, qstep, block_size=4, strength=1.0, passes=1):
    """
    Measures variance in 4x4 blocks to detect regions with many boundaries.
    Blends the original image and the smoothed image based on this local block variance.
    If passes > 1, it spatially averages the smoothing weights across adjacent blocks
    so that highly smoothed areas blend gently into unsmoothed areas.
    """
    H, W = image.shape
    smoothed_full = advanced_smoothing(image, qstep, strength)
    final_image = np.copy(image)
    
    # Estimate the expected variance from quantization noise
    noise_var = (qstep ** 2) / 12.0
    
    h_blocks = H // block_size
    w_blocks = W // block_size
    weight_map = np.zeros((h_blocks, w_blocks))
    
    # 1. Calculate the base weight map
    for y_idx in range(h_blocks):
        for x_idx in range(w_blocks):
            y = y_idx * block_size
            x = x_idx * block_size
            block = image[y:y+block_size, x:x+block_size]
            b_var = np.var(block)
            
            # Weight: 1.0 means use original (high variance = details/boundaries)
            # 0.0 means use smoothed (low variance = flat area)
            weight_map[y_idx, x_idx] = np.clip(b_var / (noise_var * 2.0 * strength), 0.0, 1.0)
            
    # 2. Spatially average the weights between adjacent blocks over multiple passes
    for _ in range(passes - 1):
        # 3x3 uniform filter averages the block's weight with its 8 neighbors
        weight_map = nd.uniform_filter(weight_map, size=3, mode='nearest')
        
    # 3. Apply the final smoothed weights to the image
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

def deblock_boundaries(image, min_block_size, base_qstep):
    """
    Applies a gradient-adaptive deblocking filter to grid boundaries.
    The filter evaluates local spatial gradients to classify boundaries and 
    applies a 3-tap normalized boxcar impulse response h[n] = [0.25, 0.5, 0.25]
    only if the gradients indicate a quantization artifact.
    """
    filtered_image = np.copy(image)
    h, w = image.shape
    
    # Dynamic thresholds based on the severity of the quantization step
    # Alpha: Max allowable cross-boundary gradient for a potential artifact
    # Beta: Max allowable internal gradient for a "flat" block (to avoid filtering textures)
    alpha = base_qstep * 1.5
    beta = base_qstep * 0.5
    
    # 1. Filter vertical boundaries
    for j in range(min_block_size, w, min_block_size):
        # Extract pixel columns around the boundary
        p1 = filtered_image[:, j-2]
        p0 = filtered_image[:, j-1] # Left of boundary
        q0 = filtered_image[:, j]   # Right of boundary
        q1 = filtered_image[:, j+1]
        
        # Calculate spatial gradients
        cross_grad = np.abs(p0 - q0)
        int_p = np.abs(p1 - p0)
        int_q = np.abs(q1 - q0)
        
        # Artifact Mask: True if it's a flat region with a moderate discontinuity
        # We only filter if the jump is significant enough to be an artifact, 
        # but not so large it's a structural edge, and the area is flat.
        artifact_mask = (cross_grad < alpha) & (int_p < beta) & (int_q < beta)
        
        # Apply filter conditionally using the mask
        # 75/25 split for the boundary pixels
        filtered_image[:, j-1] = np.where(artifact_mask, 0.75 * p0 + 0.25 * q0, p0)
        filtered_image[:, j]   = np.where(artifact_mask, 0.25 * p0 + 0.75 * q0, q0)
        
    # 2. Filter horizontal boundaries
    for i in range(min_block_size, h, min_block_size):
        # Extract pixel rows around the boundary
        p1 = filtered_image[i-2, :]
        p0 = filtered_image[i-1, :] # Top of boundary
        q0 = filtered_image[i, :]   # Bottom of boundary
        q1 = filtered_image[i+1, :]
        
        # Calculate spatial gradients
        cross_grad = np.abs(p0 - q0)
        int_p = np.abs(p1 - p0)
        int_q = np.abs(q1 - q0)
        
        # Artifact Mask: True if it's a flat region with a moderate discontinuity
        artifact_mask = (cross_grad < alpha) & (int_p < beta) & (int_q < beta)
        
        # Apply filter conditionally using the mask
        # 75/25 split for the boundary pixels
        filtered_image[i-1, :] = np.where(artifact_mask, 0.75 * p0 + 0.25 * q0, p0)
        filtered_image[i, :]   = np.where(artifact_mask, 0.25 * p0 + 0.75 * q0, q0)
        
    return filtered_image

# ==========================================
# 2. QUADTREE ENCODER / DECODER
# ==========================================
# Uses cued_sf2_lab primitives for each pipeline stage:
#   dct_ii / colxfm      – per-block DCT at the leaf's own size
#   quant1 / quant2      – adaptive quantisation
#   diagscan / runampl   – coefficient scanning
#   huffdflt / huffgen / huffenc – entropy coding
from cued_sf2_lab.dct import dct_ii, colxfm
from cued_sf2_lab.laplacian_pyramid import quant1, quant2
from cued_sf2_lab.jpeg import diagscan, runampl, huffdflt, huffgen, huffenc

MAX_BLOCK = 32    # largest leaf block (flat regions)
MIN_BLOCK = 8     # smallest leaf block (detailed regions)
MAX_DEPTH = int(np.log2(MAX_BLOCK // MIN_BLOCK))  # 2 levels of splitting
DCBITS = 10       # fixed-length DC word (safe for block sizes 8-32 with qstep>10)
DEPTH_SCALE = 1.3  # QP multiplier per depth level (> 1 = harsher for detail)


def block_edge_density(block):
    """
    Measures the density of strong edges in a block.
    Returns the percentage of pixels that exceed an edge threshold.
    This prevents splitting just because of a single noisy pixel.
    """
    if block.shape[0] < 2 or block.shape[1] < 2:
        return 0.0
    gx = np.abs(np.diff(block, axis=1))
    gy = np.abs(np.diff(block, axis=0))
    # Threshold for what constitutes a "real" edge pixel
    edge_thresh = 5.0
    num_edges = np.sum(gx > edge_thresh) + np.sum(gy > edge_thresh)
    total_pixels = gx.size + gy.size
    return num_edges / total_pixels

def deadzone_quant(x, step):
    """Deadzone quantizer: widens the zero bin to create longer runs of zeros."""
    return np.trunc(x / step)

def deadzone_dequant(xq, step):
    """Inverse deadzone quantizer (reconstructs at the bin center)."""
    return np.where(xq == 0, 0, np.sign(xq) * (np.abs(xq) + 0.5) * step)

def qt_encode(image, base_qstep, var_thresh):
    """
    Encoder pipeline:
      Image
        -> Quadtree segmentation  (variance threshold)
        -> Per-block DCT          (block size = leaf size, via dct_ii + colxfm)
        -> Adaptive quantisation  (depth-guided QP, via quant1)
        -> Entropy coding         (diagscan + runampl + Huffman)
        -> Bitstream
    """
    H, W = image.shape
    tree_bits = []     # 1 = split, 0 = leaf  (1 bit each)
    vlc_list = []      # VLC arrays for each leaf block

    # Huffman setup — default JPEG luminance tables
    huffhist = np.zeros(16 ** 2)
    dhufftab = huffdflt(1)
    _, ehuf = huffgen(dhufftab)

    def _encode_leaf(block, size, depth):
        """Single leaf: DCT -> quantise -> scan -> run-amplitude -> Huffman."""
        # 1. Forward DCT at the leaf's own block size
        C = dct_ii(size)
        Y = colxfm(colxfm(block, C).T, C).T

        # 2. Adaptive quantisation with Deadzone
        q = base_qstep * (DEPTH_SCALE ** depth)
        Yq = deadzone_quant(Y, q).astype(int)

        # 3. DC coefficient (fixed-length word)
        yqflat = Yq.flatten('F')
        dccoef = int(yqflat[0]) + 2 ** (DCBITS - 1)
        if dccoef < 0 or dccoef >= 2 ** DCBITS:
            raise ValueError(
                f'DC coeff {yqflat[0]} out of range for {DCBITS} bits '
                f'(block {size}x{size}, depth={depth}, q={q:.2f})')
        vlc_list.append(np.array([[dccoef, DCBITS]]))

        # 4. AC coefficients: diagonal scan -> run-amplitude -> Huffman
        scan = diagscan(size)
        ra = runampl(yqflat[scan])
        vlc_list.append(huffenc(huffhist, ra, ehuf))

    def _split(x, y, size, depth):
        """Recursively segment the image; encode each leaf block."""
        block = image[y:y+size, x:x+size]
        if size > MIN_BLOCK and depth < MAX_DEPTH and block_edge_density(block) > var_thresh:
            tree_bits.append(1)   # split node
            half = size // 2
            for dx, dy in [(0, 0), (half, 0), (0, half), (half, half)]:
                _split(x + dx, y + dy, half, depth + 1)
        else:
            tree_bits.append(0)   # leaf node
            _encode_leaf(block, size, depth)

    # Process each top-level macro-block
    for y in range(0, H, MAX_BLOCK):
        for x in range(0, W, MAX_BLOCK):
            _split(x, y, MAX_BLOCK, 0)

    # Assemble bitstream
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc_list)
    tree_overhead = len(tree_bits)           # 1 bit per tree node
    coeff_bits   = int(np.sum(vlc[:, 1]))    # Huffman-coded coefficients
    total_bits   = coeff_bits + tree_overhead

    header = {
        'tree': tree_bits,
        'base_qstep': base_qstep,
        'var_thresh': var_thresh,
        'tree_overhead': tree_overhead,
        'coeff_bits': coeff_bits,
    }
    return vlc, header, total_bits


def qt_decode(vlc, header, image_shape):
    """
    Decoder pipeline:
      Bitstream
        -> Entropy decode         (Huffman -> run-amplitude -> coefficients)
        -> Dequantise             (depth-guided QP, via quant2)
        -> Inverse DCT            (per-block at leaf size, via dct_ii + colxfm)
        -> Reconstruct from quadtree
        -> Deblocking filter
        -> Output image
    """
    H, W = image_shape
    out_image = np.zeros(image_shape)
    tree_iter = iter(header['tree'])
    base_qstep = header['base_qstep']

    # Huffman decoder setup
    dhufftab = huffdflt(1)
    huffcode_arr, ehuf = huffgen(dhufftab)
    huffstart = np.cumsum(np.block([0, dhufftab.bits[:15]]))
    k_powers = 2 ** np.arange(17)
    eob = ehuf[0]
    run16 = ehuf[15 * 16]

    i_ptr = [0]  # mutable pointer into vlc stream

    def _decode_leaf(x, y, size, depth):
        """Decode one leaf block from the VLC stream."""
        i = i_ptr[0]
        q = base_qstep * (DEPTH_SCALE ** depth)
        scan = diagscan(size)
        yq = np.zeros(size ** 2)

        # 1. Decode DC coefficient (fixed-length word)
        yq[0] = vlc[i, 0] - 2 ** (DCBITS - 1)
        i += 1

        # 2. Decode AC coefficients (Huffman -> run-amplitude -> values)
        cf = 0
        while np.any(vlc[i] != eob):
            run = 0
            # Decode any runs of 16 zeros
            while np.all(vlc[i] == run16):
                run += 16
                i += 1
            # Decode run and size
            start = int(huffstart[int(vlc[i, 1]) - 1])
            code_idx = start + int(vlc[i, 0]) - int(huffcode_arr[start])
            res = int(dhufftab.huffval[code_idx])
            run += res // 16
            cf += run + 1
            si = res % 16
            i += 1
            # Decode amplitude
            ampl = vlc[i, 0]
            thr = k_powers[si - 1]
            yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
            i += 1
        i += 1   # skip EOB
        i_ptr[0] = i

        # 3. Dequantise with Deadzone
        Yq = yq.reshape((size, size)).T
        Zi = deadzone_dequant(Yq, q)

        # 4. Inverse DCT at the leaf's own block size
        C = dct_ii(size)
        block = colxfm(colxfm(Zi.T, C.T).T, C.T)
        out_image[y:y + size, x:x + size] = block

    def _reconstruct(x, y, size, depth):
        """Walk the tree and decode every leaf."""
        val = next(tree_iter)
        if val == 1:
            half = size // 2
            for dx, dy in [(0, 0), (half, 0), (0, half), (half, half)]:
                _reconstruct(x + dx, y + dy, half, depth + 1)
        else:
            _decode_leaf(x, y, size, depth)

    # Reconstruct from quadtree
    for y in range(0, H, MAX_BLOCK):
        for x in range(0, W, MAX_BLOCK):
            _reconstruct(x, y, MAX_BLOCK, 0)

    # Deblocking filter
    final = deblock_boundaries(out_image, MIN_BLOCK, base_qstep)
    return final

# ==========================================
# 4. COMPARISON HELPERS
# ==========================================
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import rowdec, rowint, quantise, bpp
from cued_sf2_lab.jpeg import dwtgroup

TARGET_BITS = 5 * 1024 * 8  # 5 KB = 40960 bits

def count_bits(vlc):
    """Count total bits from a vlc array."""
    return int(np.sum(vlc[:, 1]))

def print_stats(name, total_bits, image_shape):
    """Print compression statistics."""
    num_pixels = image_shape[0] * image_shape[1]
    raw_bits = num_pixels * 8  # 8 bits per pixel for original
    ratio = raw_bits / total_bits
    bits_per_pixel = total_bits / num_pixels
    print(f"\n--- {name} ---")
    print(f"  Total bits:        {total_bits}")
    print(f"  Size:              {total_bits / 8 / 1024:.2f} KB")
    print(f"  Compression ratio: {ratio:.2f}:1")
    print(f"  Bits per pixel:    {bits_per_pixel:.3f}")
    return total_bits

def rms_error(original, reconstructed):
    """Compute RMS error between two images."""
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def ssim_score(original, compressed):
    """
    Pure SciPy/NumPy implementation of SSIM (Structural Similarity Index).
    Approximates skimage's implementation using uniform filters.
    """
    data_range = 255.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 11x11 uniform filter
    size = 11
    
    mu1 = nd.uniform_filter(original, size=size)
    mu2 = nd.uniform_filter(compressed, size=size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = nd.uniform_filter(original ** 2, size=size) - mu1_sq
    sigma2_sq = nd.uniform_filter(compressed ** 2, size=size) - mu2_sq
    sigma12 = nd.uniform_filter(original * compressed, size=size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

# --- Quadtree: binary search for base_qstep to hit target bits ---
def find_qt_params(X, target_bits=TARGET_BITS):
    """Find base_qstep that makes the quadtree encoder hit the target bit budget."""
    H, W = X.shape
    # Edge threshold: 75th percentile of 16x16 sub-block edge densities
    block_edges = [block_edge_density(X[i:i+16, j:j+16])
                  for i in range(0, H, 16) for j in range(0, W, 16)]
    var_thresh = np.percentile(block_edges, 75)

    lo, hi = 1.0, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        _, _, bits = qt_encode(X, mid, var_thresh)
        if bits > target_bits:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, var_thresh

# --- Standard DCT encoder/decoder targeting a bit budget ---
def dct_encode_decode(X, target_bits=TARGET_BITS, N=8):
    """Encode with jpegenc (standard DCT), search for qstep to hit target."""
    lo, hi = 0.5, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        vlc, htab = jpegenc(X, mid, N=N, log=False)
        bits = count_bits(vlc)
        if bits > target_bits:
            lo = mid
        else:
            hi = mid
    qstep = (lo + hi) / 2
    vlc, htab = jpegenc(X, qstep, N=N, log=False)
    Z = jpegdec(vlc, qstep, N=N, log=False)
    return Z, count_bits(vlc), qstep

# --- LBT encoder/decoder targeting a bit budget ---
def lbt_encode_decode(X, target_bits=TARGET_BITS, N=8, s=None):
    """Encode with LBT (pre-filter + DCT via jpegenc), search for qstep."""
    if s is None:
        s = (1 + 5**0.5) / 2  # golden ratio
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N//2:-N//2]

    # Forward pre-filter only (jpegenc will apply DCT internally)
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    # Binary search for qstep — jpegenc does DCT + quantize + Huffman
    lo, hi = 0.5, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        vlc, htab = jpegenc(Xp, mid, N=N, log=False)
        bits = count_bits(vlc)
        if bits > target_bits:
            lo = mid
        else:
            hi = mid
    qstep = (lo + hi) / 2
    vlc, htab = jpegenc(Xp, qstep, N=N, log=False)

    # jpegdec returns inverse-DCT'd + dequantised image, i.e. the pre-filtered domain
    Zp = jpegdec(vlc, qstep, N=N, log=False)

    # Inverse pre-filter
    Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
    Zp[t, :] = colxfm(Zp[t, :], Pr.T)

    total_bits = count_bits(vlc)
    return Zp, total_bits, qstep

# --- DWT encoder/decoder targeting a bit budget ---
def dwt_encode_decode(X, target_bits=TARGET_BITS, n_levels=3):
    """Encode with multi-level DWT, direct quantisation, and entropy (bpp) bit counting."""
    # Forward DWT
    m = X.shape[0]
    Y = X.copy()
    for i in range(n_levels):
        m_cur = m // (2**i)
        Y[:m_cur, :m_cur] = dwt(Y[:m_cur, :m_cur])

    # Helper to split DWT coefficients into subbands for independent bit counting
    def _dwt_subbands(arr, levels):
        bands = []
        for lev in range(levels):
            m_lev = arr.shape[0] // (2**(lev+1))
            bands.append(arr[:m_lev, m_lev:2*m_lev])   # LH
            bands.append(arr[m_lev:2*m_lev, :m_lev])    # HL
            bands.append(arr[m_lev:2*m_lev, m_lev:2*m_lev])  # HH
        m_lev = arr.shape[0] // (2**levels)
        bands.append(arr[:m_lev, :m_lev])  # LL (coarsest)
        return bands

    def _dwt_total_bits(Y, qstep):
        Yq = quantise(Y, qstep)
        bands = _dwt_subbands(Yq, n_levels)
        return sum(bpp(b) * b.size for b in bands)

    # Binary search for qstep
    lo, hi = 0.5, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _dwt_total_bits(Y, mid) > target_bits:
            lo = mid
        else:
            hi = mid
    qstep = (lo + hi) / 2

    # Quantise and reconstruct
    Z = quantise(Y, qstep)
    total_bits = int(_dwt_total_bits(Y, qstep))

    # Inverse DWT
    m = X.shape[0]
    for i in range(n_levels - 1, -1, -1):
        m_cur = m // (2**i)
        Z[:m_cur, :m_cur] = idwt(Z[:m_cur, :m_cur])

    return Z, total_bits, qstep

# --- Laplacian Pyramid encoder/decoder targeting a bit budget ---
def pyramid_encode_decode(X, target_bits=TARGET_BITS, n_levels=3):
    """Encode with Laplacian Pyramid, search for qstep to hit target bits."""
    h = np.array([1, 2, 1]) / 4.0  # simple binomial filter

    # Build pyramid
    layers = []
    current = X.copy()
    for i in range(n_levels):
        decimated = rowdec(rowdec(current, h).T, h).T
        interpolated = rowint(rowint(decimated, 2*h).T, 2*h).T
        diff = current - interpolated
        layers.append(diff)
        current = decimated
    layers.append(current)  # coarsest level

    # Binary search for qstep
    lo, hi = 0.5, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        total_bits = 0
        for layer in layers:
            q = quantise(layer, mid)
            total_bits += bpp(q) * q.size
        if total_bits > target_bits:
            lo = mid
        else:
            hi = mid
    qstep = (lo + hi) / 2

    # Quantise and reconstruct
    q_layers = [quantise(layer, qstep) for layer in layers]
    total_bits = sum(bpp(q) * q.size for q in q_layers)

    # Inverse pyramid
    recon = q_layers[-1]
    for i in range(n_levels - 1, -1, -1):
        interpolated = rowint(rowint(recon, 2*h).T, 2*h).T
        recon = q_layers[i] + interpolated

    return recon, int(total_bits), qstep


# ==========================================
# 5. MAIN - RUN ALL COMPARISONS
# ==========================================
if __name__ == '__main__' or True:
    test_images = ['lighthouse.mat', 'bridge.mat', 'flamingo.mat']
    
    for img_name in test_images:
        print("\n\n" + "*" * 70)
        print(f"*** PROCESSING {img_name} ***")
        print("*" * 70)

        # Load image (ignore colormaps since not all files have them)
        X, _ = load_mat_img(img_name, 'X', [])
        X = X - 128.0
        H, W = X.shape
        print(f"Image size: {H}x{W} = {H*W} pixels")
        print(f"Raw size:   {H*W*8} bits = {H*W*8/8/1024:.1f} KB")
        print(f"Target:     {TARGET_BITS} bits = {TARGET_BITS/8/1024:.1f} KB")
        print("=" * 55)

        results = {}  # name -> (reconstructed, total_bits, rms, ssim)

        methods = [
            "Quadtree (custom)",
            "Quadtree (Smoothed)",
            "Quadtree (4x4 Adaptive)",
            "DCT 8x8",
            "LBT 8x8",
            "DWT 3-level",
            "Lap. Pyramid"
        ]
        
        qt_cache = None

        for name in methods:
            print(f"\n[Running] {name}...")
            if name == "Quadtree (custom)":
                qt_qstep, qt_vthresh = find_qt_params(X)
                qt_vlc, qt_header, qt_bits = qt_encode(X, qt_qstep, qt_vthresh)
                recon = qt_decode(qt_vlc, qt_header, X.shape)
                bits = qt_bits
                qstep = qt_qstep
                qt_cache = (recon, bits, qstep) # Cache for smooth variants
                print(f"  [Info] tree overhead: {qt_header['tree_overhead']} bits, leaf blocks: {qt_header['tree'].count(0)}")
            elif name == "Quadtree (Smoothed)":
                recon = advanced_smoothing(qt_cache[0], qt_cache[2])
                bits, qstep = qt_cache[1], qt_cache[2]
            elif name == "Quadtree (4x4 Adaptive)":
                recon = block_adaptive_smoothing(qt_cache[0], qt_cache[2], block_size=4)
                bits, qstep = qt_cache[1], qt_cache[2]
            elif name == "DCT 8x8":
                recon, bits, qstep = dct_encode_decode(X)
            elif name == "LBT 8x8":
                recon, bits, qstep = lbt_encode_decode(X)
            elif name == "DWT 3-level":
                recon, bits, qstep = dwt_encode_decode(X, n_levels=3)
            elif name == "Lap. Pyramid":
                recon, bits, qstep = pyramid_encode_decode(X, n_levels=3)
                
            rms = rms_error(X, recon)
            ssim_val = ssim_score(X, recon)
            print_stats(name, bits, X.shape)
            print(f"  RMS error:         {rms:.2f}")
            print(f"  SSIM:              {ssim_val:.4f}")
            print(f"  qstep used:        {qstep:.2f}")
            results[name] = (recon, bits, rms, ssim_val)

        # ==========================================
        # 6. SUMMARY TABLE
        # ==========================================
        print("\n" + "=" * 65)
        print(f"{'Method':<24} {'Bits':>8} {'KB':>7} {'BPP':>7} {'RMS':>7} {'SSIM':>8}")
        print("-" * 65)
        for name, (recon, bits, rms, ssim_val) in results.items():
            kb = bits / 8 / 1024
            bpp_val = bits / (H * W)
            print(f"{name:<24} {bits:>8} {kb:>7.2f} {bpp_val:>7.3f} {rms:>7.2f} {ssim_val:>8.4f}")
        print("=" * 65)

        # ==========================================
        # 7. PLOT ALL RESULTS
        # ==========================================
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Original
        plot_image(X + 128.0, ax=axes[0, 0])
        axes[0, 0].set_title(f'Original ({img_name})', fontsize=11)

        for idx, (name, (recon, bits, rms, ssim_val)) in enumerate(results.items()):
            row = (idx + 1) // 4
            col = (idx + 1) % 4
            plot_image(recon + 128.0, ax=axes[row, col])
            axes[row, col].set_title(f'{name}\n{bits/8/1024:.2f}KB\nRMS: {rms:.1f} | SSIM: {ssim_val:.3f}', fontsize=10)

        fig.suptitle(f'{img_name}: Compression Comparison @ {TARGET_BITS/8/1024:.0f} KB Target', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show(block=True)
        
        # ==========================================
        # 8. OPTIMIZATION LOOP PLOT FOR SMOOTHING
        # ==========================================
        qt_recon, qt_bits, qt_qstep = qt_cache
        strengths = [0.5, 1.0, 1.5]
        
        fig_opt, axes_opt = plt.subplots(3, 4, figsize=(20, 15))
        fig_opt.suptitle(f"Smoothing Optimization Passes - {img_name}", fontsize=16)
        
        # Row 0: Advanced Smoothing (varying strength)
        plot_image(qt_recon + 128.0, ax=axes_opt[0, 0])
        axes_opt[0, 0].set_title(f"Base Quadtree\nSSIM: {results['Quadtree (custom)'][3]:.4f}")
        axes_opt[0, 0].axis('off')
        
        for idx, s in enumerate(strengths):
            sm = advanced_smoothing(qt_recon, qt_qstep, strength=s)
            ssim_val = ssim_score(X, sm)
            plot_image(sm + 128.0, ax=axes_opt[0, idx+1])
            axes_opt[0, idx+1].set_title(f"Adv Smooth (strength={s})\nSSIM: {ssim_val:.4f}")
            axes_opt[0, idx+1].axis('off')
            
        # Row 1: Adaptive Smoothing (varying strength, passes=1)
        plot_image(qt_recon + 128.0, ax=axes_opt[1, 0])
        axes_opt[1, 0].set_title(f"Base Quadtree")
        axes_opt[1, 0].axis('off')
        
        for idx, s in enumerate(strengths):
            sm = block_adaptive_smoothing(qt_recon, qt_qstep, block_size=4, strength=s, passes=1)
            ssim_val = ssim_score(X, sm)
            plot_image(sm + 128.0, ax=axes_opt[1, idx+1])
            axes_opt[1, idx+1].set_title(f"4x4 Adapt (strength={s})\nSSIM: {ssim_val:.4f}")
            axes_opt[1, idx+1].axis('off')
            
        # Row 2: Spatial Block Averaging (fixed strength=1.0, varying spatial passes)
        spatial_passes = [1, 2, 3]
        plot_image(qt_recon + 128.0, ax=axes_opt[2, 0])
        axes_opt[2, 0].set_title(f"Base Quadtree")
        axes_opt[2, 0].axis('off')
        
        for idx, p in enumerate(spatial_passes):
            sm = block_adaptive_smoothing(qt_recon, qt_qstep, block_size=4, strength=1.0, passes=p)
            ssim_val = ssim_score(X, sm)
            plot_image(sm + 128.0, ax=axes_opt[2, idx+1])
            axes_opt[2, idx+1].set_title(f"4x4 Spatial (passes={p})\nSSIM: {ssim_val:.4f}")
            axes_opt[2, idx+1].axis('off')
            
        plt.tight_layout()
        plt.show(block=True)  # Block so images show sequentially