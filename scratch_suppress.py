import numpy as np
from frontends import DCT_8x8
from cued_sf2_lab.familiarisation import load_mat_img

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
lighthouse = lighthouse - 128.0

def suppress_dct(Yq, thresh):
    Yq_supp = Yq.copy()
    m, n = Yq.shape
    sub_m = m // 8
    sub_n = n // 8
    for u in range(8):
        for v in range(8):
            if u + v >= thresh:
                sub_image = Yq_supp[u*sub_m : (u+1)*sub_m, v*sub_n : (v+1)*sub_n]
                sub_image[:] = 0.0
    return Yq_supp

dct = DCT_8x8()
Y_dct = dct.encode(lighthouse)
step = 11.17 # base optimal step for ratio 1.0

print(f"{'Threshold':<10} | {'RMS Error':<15} | {'Bits/Pixel':<15}")
for thresh in range(15, 7, -1):
    Yq = dct.quant(Y_dct, step, 1.0)
    Yq_supp = suppress_dct(Yq, thresh)
    rms = np.std(lighthouse - dct.decode(Yq_supp))
    bits = dct.get_bits(Yq_supp)
    print(f"{thresh:<10} | {rms:<15.4f} | {bits:<15.4f}")

