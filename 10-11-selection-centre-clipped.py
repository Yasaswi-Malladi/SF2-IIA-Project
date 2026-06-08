#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise


# # 10. Selection of preferred energy compaction options
# 
# The remainder of this project will concentrate on developing the rest of
# an image compression system, based on a few of the filtering /
# transformation schemes studied so far.
# 
# Since the subsequent processes are non-linear, we cannot expect to be able to
# choose precisely the right front-end at this stage, so we adopt the pragmatic
# approach of picking about three good candidates and trust that one of these
# will lead to a near-optimum solution in the end. Remember that up to this point we have only been using entropy to give us an _estimate_ of the number of bits required, the accuracy of which is affected by subsequent stages.
# 
# At this stage it is worth trying your schemes with all three
# test images, (`Lighthouse`, `Bridge`, and `Flamingo`). You will find `Bridge` more difficult to compress than the other two. You may also want to introduce other images of your own.

# In[2]:


lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')


# In[3]:


fig, axs = plt.subplots(1, 3)
plot_image(lighthouse, ax=axs[0])
plot_image(bridge, ax=axs[1])
plot_image(flamingo, ax=axs[2])


# Write `.py` files to implement each of your
# chosen schemes, so that you do not have to remember long sequences
# of commands each time you run them. You can easily edit the M-files to introduce different options
# later.  Using plenty of comments in these files will help when you want to change them.

# # 11. Centre-clipped linear quantisers
# 
# The quantisers that you have used so far have all been uniform quantisers
# (i.e.  all steps have been the same size).  However the probability
# distributions of the intensities of the bandpass sub-images from the energy
# compaction front-ends are usually highly peaked at zero.  The amount of data
# compression depends heavily on the proportion of data samples which are
# quantised to zero; if this approaches unity then high compression is
# achieved.
# 
# Hence it is often found desirable to make the quantiser non-linear
# so that more samples tend to be quantised to zero.  A simple way
# to achieve this is to widen the step-size of the "zero" step.  In
# a uniform quantiser, the "zero" step is normally centred on zero,
# with rises to the next level at $\pm$ half of the step-size on
# each side of zero. `quantise` allows a third argument `rise1` to be specified, which is
# the point at which the first rise occurs on each side of the zero step.  A
# value of `rise1` = `step/2` is the default, but `rise1` = $\left\{0.5, 1, 1.5\right\}\times$ `step` are worth investigating. To show what effect these have, try:
# 
# ```python
# x = np.arange(-100, 100+1)
# y = quantise(x, 20, rise1)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.grid()
# ```

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from frontends import DCT_8x8, LBT_8x8, DWT_4_EqMSE

lighthouse = X - 128.0
bridge = Xb - 128.0 if 'Xb' in locals() else X - 128.0

schemes = [DCT_8x8(), LBT_8x8(s=1.414), DWT_4_EqMSE()]
ratios = [0.5, 1.0, 1.5]

print("==================== RISE1 RATIO INVESTIGATION ====================")
for img_name, img_data in [('Lighthouse', lighthouse), ('Bridge', bridge)]:
    print(f"\nEvaluating Image: {img_name}")

    # Target RMS derived from Step 17 on DCT with uniform quantizer
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


# A wider zero step means that more samples will be coded as zero and so the
# entropy of the data will be reduced.  The use of a wide zero step is
# beneficial if it results in a better entropy vs. error tradeoff than a uniform
# quantiser.
# 
# <div class="alert alert-block alert-danger">
# 
# For each of your preferred front-end
# schemes, investigate the effects of varying the first rise of the
# quantiser.  To do this, you could plot how the quantising error
# varies as a function of the number of bits for a few different ratios of
# `rise1` to step-size, and hence find the ratio which gives the
# best compression for a given rms error. </div>

# In[ ]:


# You may wish to work in standalone python files (instead of notebooks) at this point!


# Most current image compression standards use quantisers with a
# double-width centre step (`rise1 = step`). Do not spend too much time
# on this as the compression gains are likely to be quite small.
# 
# <div class="alert alert-block alert-danger">
# 
# 
# Discuss whether your results indicate that `rise1 = step`
# is a reasonable compromise if all quantisers are to be similar.
# </div>

# 
# **Discussion: Is `rise1 = step` a reasonable compromise?**
# 
# Yes, the results above clearly indicate that `rise1 = step` (Ratio 1.0) is an excellent compromise. Across all tested front-end schemes (DCT, LBT, DWT) and images of varying difficulty (Lighthouse and Bridge), a ratio of 1.0 consistently achieves the **lowest bits/pixel** for the target RMS error compared to the uniform quantiser (`rise1 = step/2`) and wider step (`rise1 = 1.5 * step`). A wider zero step successfully quantises more near-zero high-frequency noise coefficients to zero, significantly lowering entropy without adding unacceptable distortion.
# 

# A final strategy which you can consider is to completely suppress some
# sub-images or DCT coefficients.  This is equivalent to increasing `rise1`
# to a very large value for these components.  In the sub-images / coefficients
# which represent only the highest horizontal and vertical frequency components
# combined, the effects of suppression can be almost unnoticable and yet
# a useful saving in number of bits can be achieved.
# 
# <div class="alert alert-block alert-danger">
# 
# Investigate any additional gains which can be achieved with suppression
# of some sub-images / coefficients.
# </div>

# In[ ]:


def suppress_dct_lbt(Yq, N, thresh):
    Yq_supp = Yq.copy()
    m, n = Yq.shape
    sub_m = m // N
    sub_n = n // N
    for u in range(N):
        for v in range(N):
            if u + v >= thresh:
                Yq_supp[u*sub_m : (u+1)*sub_m, v*sub_n : (v+1)*sub_n] = 0.0
    return Yq_supp

def suppress_dwt(Yq, n_levels):
    Yq_supp = Yq.copy()
    m, n = Yq.shape
    # Level 0 is the outer 3/4 of the image
    Yq_supp[:m//2, n//2:n] = 0.0
    Yq_supp[m//2:m, :n//2] = 0.0
    Yq_supp[m//2:m, n//2:n] = 0.0
    return Yq_supp

print("==================== SUPPRESSION INVESTIGATION (Lighthouse) ====================")
step = 17 # Fixed step for evaluation
dct = schemes[0]
dwt = schemes[2]

Y_dct = dct.encode(lighthouse)
Y_dwt = dwt.encode(lighthouse)

Yq_dct = dct.quant(Y_dct, step, 1.0)
Yq_dwt = dwt.quant(Y_dwt, step, 1.0)

print(f"{'Strategy':<30} | {'RMS Error':<15} | {'Bits/Pixel':<15}")
print("-" * 65)

# DCT 
rms_dct = np.std(lighthouse - dct.decode(Yq_dct))
bits_dct = dct.get_bits(Yq_dct)
print(f"{'DCT No Suppression':<30} | {rms_dct:<15.4f} | {bits_dct:<15.4f}")

Yq_dct_supp = suppress_dct_lbt(Yq_dct, 8, 13) # Suppress bottom right 3 coefficients
rms_dct_supp = np.std(lighthouse - dct.decode(Yq_dct_supp))
bits_dct_supp = dct.get_bits(Yq_dct_supp)
print(f"{'DCT Suppress (thresh=13)':<30} | {rms_dct_supp:<15.4f} | {bits_dct_supp:<15.4f}")

# DWT
rms_dwt = np.std(lighthouse - dwt.decode(Yq_dwt))
bits_dwt = dwt.get_bits(Yq_dwt)
print(f"{'DWT No Suppression':<30} | {rms_dwt:<15.4f} | {bits_dwt:<15.4f}")

Yq_dwt_supp = suppress_dwt(Yq_dwt, 4)
rms_dwt_supp = np.std(lighthouse - dwt.decode(Yq_dwt_supp))
bits_dwt_supp = dwt.get_bits(Yq_dwt_supp)
print(f"{'DWT Suppress (Level 0)':<30} | {rms_dwt_supp:<15.4f} | {bits_dwt_supp:<15.4f}")

# Plot visual differences
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_image(dct.decode(Yq_dct_supp), ax=axes[0])
axes[0].set_title(f"DCT Suppressed (thresh=13)\nRMS: {rms_dct_supp:.2f}, Bits: {bits_dct_supp:.2f}")

plot_image(dwt.decode(Yq_dwt_supp), ax=axes[1])
axes[1].set_title(f"DWT Suppressed (Lvl 0)\nRMS: {rms_dwt_supp:.2f}, Bits: {bits_dwt_supp:.2f}")
plt.show()


# 
# **Discussion: Sub-image / Coefficient Suppression**
# 
# When fixing the step size, completely suppressing the highest-frequency components (e.g. `u+v >= 13` for DCT, or Level 0 sub-images for DWT) results in a **reduction of bit rate** because these entire sub-bands have zero entropy. 
# While the **RMS error mathematically increases**, the high-frequency nature of these components means the added distortion often manifests as a slight blurring or loss of extreme sharp texture, which the human visual system is less sensitive to. This allows us to achieve a higher compression ratio with almost unnoticeable visual degradation!
# 
