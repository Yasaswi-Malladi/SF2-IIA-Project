import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from frontends import DCT_8x8, LBT_8x8, DWT_4_EqMSE
from cued_sf2_lab.familiarisation import load_mat_img

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
lighthouse = lighthouse - 128.0

bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
bridge = bridge - 128.0

schemes = [DCT_8x8(), LBT_8x8(s=1.414), DWT_4_EqMSE()]
ratios = [0.5, 1.0, 1.5]

for img_name, img_data in [('Lighthouse', lighthouse), ('Bridge', bridge)]:
    print(f"\n==================== IMAGE: {img_name} ====================")
    
    # We use step=17 for uniform quantiser to find target RMS
    # Using DCT for target RMS
    dct = schemes[0]
    Y_ref = dct.encode(img_data)
    Yq_ref = dct.quant(Y_ref, 17, rise1_ratio=0.5)
    Z_ref = dct.decode(Yq_ref)
    target_rms = np.std(img_data - Z_ref)
    
    print(f"Target RMS Error: {target_rms:.4f}")
    print("{:<20} | {:<10} | {:<15} | {:<12}".format("Scheme", "Rise Ratio", "Base Step", "Bits/Pixel"))
    print("-" * 65)
    
    for scheme in schemes:
        Y = scheme.encode(img_data)
        
        for ratio in ratios:
            def check_rms(step):
                Yq = scheme.quant(Y, step, rise1_ratio=ratio)
                Z = scheme.decode(Yq)
                return (np.std(img_data - Z) - target_rms) ** 2
                
            res = minimize_scalar(check_rms, bounds=(5, 50), method='bounded')
            opt_step = res.x
            
            Yq_opt = scheme.quant(Y, opt_step, rise1_ratio=ratio)
            bits = scheme.get_bits(Yq_opt)
            
            print("{:<20} | {:<10.1f} | {:<15.2f} | {:<12.4f}".format(scheme.name, ratio, opt_step, bits))
    print("-" * 65)
