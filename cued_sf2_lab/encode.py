""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp

from .common import HeaderType

def deadzone_quant(x, step):
    return np.sign(x) * np.floor(np.abs(x) / step)

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

def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header. """
    return int(header['bits'])

def encode(X: np.ndarray) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
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
    
    target_bits = 40000
    
    def simulate(qstep):
        Yrq = deadzone_quant(Yr, qstep)
        tree_bits, blocks = get_qt_structure(Yrq, min_size=4)
        coef_bits = 0
        for (x, y, size) in blocks:
            block = Yrq[y:y+size, x:x+size]
            if not np.all(block == 0):
                coef_bits += bpp(block) * block.size
        return len(tree_bits) + coef_bits, Yrq
        
    # Bisection search to hit exactly 40000 bits
    lo, hi = 0.1, 200.0
    for _ in range(30):
        mid = (lo + hi) / 2
        bits, _ = simulate(mid)
        if bits > target_bits: 
            lo = mid
        else: 
            hi = mid
            
    qstep = (lo + hi) / 2
    final_bits, Yrq = simulate(qstep)
    
    header = {
        'Yrq': Yrq,
        'qstep': qstep,
        'bits': final_bits
    }
    
    vlc = np.zeros((0, 2), dtype=int)
    return vlc, header