import numpy as np
from frontends import DCT_8x8, LBT_8x8, DWT_4_EqMSE
from cued_sf2_lab.familiarisation import load_mat_img

X, _ = load_mat_img(img='lighthouse.mat', img_info='X')
X = X - 128.0

dct = DCT_8x8()
lbt = LBT_8x8()
dwt = DWT_4_EqMSE()

for scheme in [dct, lbt, dwt]:
    Y = scheme.encode(X)
    Yq = scheme.quant(Y, 17, rise1_ratio=0.5)
    Z = scheme.decode(Yq)
    bits = scheme.get_bits(Yq)
    rms = np.std(X - Z)
    print(f"{scheme.name}: RMS = {rms:.4f}, bits = {bits:.4f}")
