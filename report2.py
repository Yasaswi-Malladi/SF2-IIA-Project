from Image import *

lighthouse=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]- 128.0
bridge=load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})[0]-128.0

def pyramid_optimisation():
    im = Image(lighthouse)
    reference = im.quantise(17, False)
    h = [.25, .5, .25]
    step, r = im.comp_ratio_pyramid_mse(h, 2, 17)
    print(step, r)
    im.pyramid(h, 2)
    im.quantise_pyramid([step]*3) #not correct
    im.ipyramid(h)
    print(im.rms(lighthouse))
    print(reference.rms(lighthouse))
    im.plot()


def DCT_visualisation():
    im = Image(lighthouse)
    im.DCT(8)
    Image(im.regroup_N(8)).plot()
# DCT_visualisation()

def DCT_quantisation():
    for n in [4, 8, 16]:
        im = Image(lighthouse)
        im.DCT(n)
        print(im.bits_dct(n))
# DCT_quantisation()

def DCT_optimisation():
    N = 8
    im = Image(lighthouse)
    step, r = im.comp_ratio_dct(N, 17)
    print(step, r)
    im.DCT(N)
    im.quantise(step)
    im.iDCT(N)
    im.plot()
# DCT_optimisation()

def DCT_compresion(X):
    im = Image(lighthouse)
    r, x = im.comp_ratio_dct(8, 17)
    im.DCT(8)
    im.quantise(x)
    print(x, r)

# im = Image(lighthouse)
# h = [.25, .5, .25]
# im.pyramid(h, 4)
# im.quantise_pyramid([30, 30, 30, 30, 30])
# im.ipyramid(h)
# im.plot()

# im = Image(lighthouse)
# print(len(lighthouse))
# M = np.array([
#     [10, 10, 10, 30],
#     [10, 10, 10, 0],
#     [10, 10, 10, 0]
# ])
# # im.plot()
# im.DWT(l=3)
# im.plot()
# im.quantise_dwt(M ,3)
# im.iDWT(l=3)
# im.plot()

# h = [.25, .5, .25]
# print(Image.MSE_pyramid(h, 4))

def DCT_compression_ratios(X):
    '''Repeat the main measurements from the previous section, so as to obtain estimates of the number of bits and compression 
    ratios for 4 × 4 and 16 × 16 DCTs when the rms errors are equivalent to those in your previous tests. Also assess the relative
      subjective quality of the reconstructed images'''
    w, h = X.shape
    fig, ax = plt.subplots(3, 3, figsize=(3*w/100, 3*h/100))
    
    for yi, ref_step in enumerate([17, 30, 50]):
        for ni, n in enumerate([4, 8, 16]):
            im = Image(X)
            h_opt, r = im.comp_ratio_dct(n, ref_step)
            im.DCT(n)
            im.quantise(h_opt)
            im.iDCT(n)
            plot_image(im.image, ax=ax[yi, ni])
            ax[yi, ni].axis("off")
            # ax.set_title(f"Ratio")
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()

def DCT_compression_ratios_3(X):
    '''Repeat the main measurements from the previous section, so as to obtain estimates of the number of bits and compression 
    ratios for 4 × 4 and 16 × 16 DCTs when the rms errors are equivalent to those in your previous tests. Also assess the relative
      subjective quality of the reconstructed images'''
    w, h = X.shape
    fig, ax = plt.subplots(1, 3, figsize=(3*w/100, h/100))
    
    for ni, n in enumerate([4, 8, 16]):
        im = Image(X)
        h_opt, r = im.comp_ratio_dct(n, 17)
        im.DCT(n)
        im.quantise(h_opt)
        im.iDCT(n)
        plot_image(im.image, ax=ax[ni])
        ax[ni].axis("off")
        # ax.set_title(f"Ratio")
        print(r)
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()
# DCT_compression_ratios_3(lighthouse)

def LBT_compression_ratios(X):
    '''Investigate the relative visual and compression performance of LBTs with 4 × 4, 8 × 8 and 16 × 16 blocks, using the 
    scaling factor you have previously selected. As before, be careful to match the rms error with a directly quantised image.
    For an 8 × 8 DCT, try implementing an LBT with POT scaling factors varying from 1 to 2 (√2 is often a good choice). 
    In each case find the quantisation step which makes the rms error match the directly quantised image. Note the compression 
    ratios and find the scaling factor which maximises these. Also note the visual features in these images.'''
    w, h = X.shape
    fig, ax = plt.subplots(1, 5, figsize=(5*w/100, h/100))
    n = 8
    
    for si, s in enumerate([1,2**0.5, 1.5, (1 + (5 ** 0.5)) / 2, 2]):
        im = Image(X)
        im.POT(n, s)
        h_opt, r = im.comp_ratio_lbt(n,s, 17)
        im.quantise(h_opt)
        im.iPOT(n)
        plot_image(im.image, ax=ax[si])
        ax[si].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()
# LBT_compression_ratios(lighthouse)  

def lbt_reconstruction(X):
    w, h = X.shape
    fig, ax = plt.subplots(1, 3, figsize=(3*w/100, h/100))
    n = 8
    s = pow(2, 0.5)
    for ni, n in enumerate([4, 8, 16]):
        im = Image(X)
        im.POT(n, s)
        h_opt, r = im.comp_ratio_lbt(n,s, 17)
        im.quantise(h_opt)
        im.iPOT(n)
        plot_image(im.image, ax=ax[ni])
        ax[ni].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()
# lbt_reconstruction(lighthouse)

def inner_workings_of_dwt(X):
    w, h = X.shape
    fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100))
    im = Image(X)
    im.DWT_example()
    im.image[:128, 128:] = im.image[:128, 128:]*4
    im.image[128:, :]= im.image[128:, :]*4
    plot_image(im.image, ax=ax)
    ax.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()
# inner_workings_of_dwt(lighthouse)

def dwt_const_mse(X):
    '''Investigate the performance of both an equal-step-size and an equal-MSE scheme (follow a similar procedure as you used 
for the Laplacian Pyramid to find the appropriate step- size ratios). Hence determine how many levels of DWT are reasonably
 optimal for the Lighthouse and Bridge images. Also evaluate the subjective quality of your reconstructed images, and comment
   on how this depends on n and on the way that step-sizes are assigned to the different levels. Once again, for each image
     choose quantisation steps such that you match the rms error to that for direct quantisation with a step-size of 17.'''
    w, h = X.shape
    fig, ax = plt.subplots(2, 6, figsize=(6*w/100, 2*h/100))

    for l in range(1, 7):
        im = Image(X)
        step_opt, r = im.comp_ratio_dwt(l=l)
        im.DWT(l=l)
        im.quantise_dwt(step_opt*np.ones((3, l+1)), l=l)
        im.iDWT(l=l)
        plot_image(im.image, ax=ax[0, l-1])
        ax[0, l-1].axis("off")
        ax[0, l-1].set_title(f"Const. step: {l} layer(s)")
        print(f"Compression ratio {r:.2f} | Step {step_opt} | RMS {im.rms(X)}")

    for l in range(1, 7):
        im = Image(X)
        step_opt, r = im.comp_ratio_dwt_mse(l=l)
        im.DWT(l=l)
        im.quantise_dwt(step_opt, l=l)
        im.iDWT(l=l)
        plot_image(im.image, ax=ax[1, l-1])
        ax[1, l-1].axis("off")
        ax[1, l-1].set_title(f"MSE. step: {l} layer(s)")
        print(f"Compression ratio {r:.2f} | Step {step_opt} | RMS {im.rms(X)}")
        # ax[1, l-1].set_ylabel("Label", fontsize=12, fontweight='bold', rotation=90, labelpad=10)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()

dwt_const_mse(lighthouse)

# im = Image(lighthouse)
# im.DWT(l=3)
# M = np.array(Image.MSE_dwt(h1, h2, g1, g2, l=3))
# print(M)
# im.quantise_dwt(M*0.1, l=3)
# im.iDWT(l=3)
# print(im.rms(lighthouse))

# im = Image(lighthouse)
# l=4
# im.DWT(l=l)
# step_opt, r = im.comp_ratio_dwt(l=l)
# im.quantise_dwt(step_opt*np.ones((3, l+1)), l=l)
# im.iDWT(l=l)
# # plot_image(im.image, ax=ax[0, l-1])
# # ax[0, l-1].axis("off")
# # ax[0, l-1].set_title(f"Const. step: {l} layer(s)")
# print(f"Compression ratio {r:.2f} | Step {step_opt} | RMS {im.rms(X)}")
