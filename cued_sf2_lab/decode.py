""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from scipy.ndimage import gaussian_filter

from common import HeaderType

def deadzone_dequant(xq, step):
    # Reverting to lower-edge reconstruction because AC coefficients are Laplacian distributed, 
    # meaning they are heavily clustered near the lower edge of the bin!
    return xq * step

def decode(vlc: np.ndarray, header: HeaderType) -> np.ndarray:
    """
    Parameters:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
        
    Outputs:
        X: the decoded grayscale image
    """
    Yrq = header['Yrq']
    qstep = header['qstep']
    
    # 1. Dequantize
    Yr_hat = deadzone_dequant(Yrq, qstep)
    
    # 2. Inverse Regroup
    Y_hat = regroup(Yr_hat, Yr_hat.shape[0] // 8) 
    
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
    
    # Add back the DC offset
    return X_hat + 128.0