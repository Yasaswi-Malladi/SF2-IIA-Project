# pyrefly: ignore [missing-import]
import numpy as np
from cued_sf2_lab.dct import dct_ii, colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import quantise, bpp

def nlevdwt(X, n):
    Y = X.copy()
    m, n_dim = Y.shape
    for i in range(n):
        Y[:m, :n_dim] = dwt(Y[:m, :n_dim])
        m //= 2
        n_dim //= 2
    return Y

def nlevidwt(Y, n):
    X = Y.copy()
    m, n_dim = Y.shape
    m //= (2 ** (n - 1))
    n_dim //= (2 ** (n - 1))
    for i in range(n):
        X[:m, :n_dim] = idwt(X[:m, :n_dim])
        m *= 2
        n_dim *= 2
    return X

def dctbpp(Yr, N):
    m, n = Yr.shape
    sub_m = m // N
    sub_n = n // N
    total_bits = 0.0
    for u in range(N):
        for v in range(N):
            sub_image = Yr[u*sub_m : (u+1)*sub_m, v*sub_n : (v+1)*sub_n]
            total_bits += bpp(sub_image) * sub_image.size
    return total_bits / Yr.size

def get_dwt_bpp(dwtent):
    n = dwtent.shape[1] - 1
    total_bpp = 0.0
    for i in range(n):
        weight = 1.0 / (4 ** (i + 1))
        total_bpp += (dwtent[0, i] + dwtent[1, i] + dwtent[2, i]) * weight
    total_bpp += dwtent[0, n] * (1.0 / (4 ** n))
    return total_bpp

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

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray, rise1_ratio=0.5):
    n = dwtstep.shape[1] - 1
    Yq = Y.copy()
    dwtent = np.zeros_like(dwtstep)
    height, width = Y.shape
    for i in range(n):
        m_h = height // (2**i)
        m_w = width // (2**i)
        
        tr = Yq[:m_h//2, m_w//2:m_w]
        bl = Yq[m_h//2:m_h, :m_w//2]
        br = Yq[m_h//2:m_h, m_w//2:m_w]
        
        s = dwtstep[0, i]
        tr_q = quantise(tr, s, s * rise1_ratio)
        Yq[:m_h//2, m_w//2:m_w] = tr_q
        dwtent[0, i] = bpp(tr_q)
        
        s = dwtstep[1, i]
        bl_q = quantise(bl, s, s * rise1_ratio)
        Yq[m_h//2:m_h, :m_w//2] = bl_q
        dwtent[1, i] = bpp(bl_q)
        
        s = dwtstep[2, i]
        br_q = quantise(br, s, s * rise1_ratio)
        Yq[m_h//2:m_h, m_w//2:m_w] = br_q
        dwtent[2, i] = bpp(br_q)
        
    m_h = height // (2**(n-1))
    m_w = width // (2**(n-1))
    lp = Yq[:m_h//2, :m_w//2]
    
    s = dwtstep[0, n]
    lp_q = quantise(lp, s, s * rise1_ratio)
    Yq[:m_h//2, :m_w//2] = lp_q
    dwtent[0, n] = bpp(lp_q)
    
    return Yq, dwtent

class DCT_8x8:
    def __init__(self):
        self.N = 8
        self.C = dct_ii(8)
        self.name = "DCT 8x8"
        
    def encode(self, X):
        return colxfm(colxfm(X, self.C).T, self.C).T
        
    def quant(self, Y, step, rise1_ratio=0.5):
        rise1 = step * rise1_ratio
        return quantise(Y, step, rise1)
        
    def decode(self, Yr):
        return colxfm(colxfm(Yr, self.C.T).T, self.C.T).T
        
    def get_bits(self, Yr):
        Yr_grouped = self._regroup(Yr, self.N)
        return dctbpp(Yr_grouped, self.N)
        
    def _regroup(self, X, N):
        M = X.shape[0] // N
        Y = np.zeros_like(X)
        for i in range(N):
            for j in range(N):
                Y[i*M:(i+1)*M, j*M:(j+1)*M] = X[i::N, j::N]
        return Y

class LBT_8x8:
    def __init__(self, s=1.414):
        self.N = 8
        self.s = s
        self.Pf, self.Pr = pot_ii(8, s)
        self.C = dct_ii(8)
        self.t = np.s_[4:-4]
        self.name = f"LBT 8x8 (s={s:.3f})"
        
    def encode(self, X):
        Xp = X.copy()
        Xp[self.t, :] = colxfm(Xp[self.t, :], self.Pf)
        Xp[:, self.t] = colxfm(Xp[:, self.t].T, self.Pf).T
        return colxfm(colxfm(Xp, self.C).T, self.C).T
        
    def quant(self, Y, step, rise1_ratio=0.5):
        rise1 = step * rise1_ratio
        return quantise(Y, step, rise1)
        
    def decode(self, Yr):
        Z = colxfm(colxfm(Yr, self.C.T).T, self.C.T).T
        Zp = Z.copy()
        Zp[:, self.t] = colxfm(Zp[:, self.t].T, self.Pr.T).T
        Zp[self.t, :] = colxfm(Zp[self.t, :], self.Pr.T)
        return Zp
        
    def get_bits(self, Yr):
        Yr_grouped = self._regroup(Yr, 16)
        return dctbpp(Yr_grouped, 16)
        
    def _regroup(self, X, N):
        M = X.shape[0] // N
        Y = np.zeros_like(X)
        for i in range(N):
            for j in range(N):
                Y[i*M:(i+1)*M, j*M:(j+1)*M] = X[i::N, j::N]
        return Y

class DWT_4_EqMSE:
    def __init__(self):
        self.n = 4
        self.name = "DWT 4-Level Eq-MSE"
        self._last_dwtent = None
        
    def encode(self, X):
        return nlevdwt(X, self.n)

    def quant(self, Y, base_step, rise1_ratio=0.5):
        steps = get_dwt_equal_mse_steps(base_step, self.n)
        Yq, dwtent = quantdwt(Y, steps, rise1_ratio)
        self._last_dwtent = dwtent
        return Yq
        
    def decode(self, Yr):
        return nlevidwt(Yr, self.n)
        
    def get_bits(self, Yr):
        # uses the entropy calculated during quantization
        return get_dwt_bpp(self._last_dwtent)
