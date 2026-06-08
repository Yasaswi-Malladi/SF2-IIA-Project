from Image import *
import matplotlib.pyplot as plt

lighthouse=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]-128.0
bridge=load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})[0]-128.0
flamingo=load_mat_img(img='flamingo.mat', img_info='X')[0]-128.0
# ---
# testing different quantisation methods
def histogram_increasing_spacing(image):
    im = Image(image)
    
    im1 = im.quantise(17)
    im2 = im.quantise_exponential(20, 0.5)
    im3 = im.quantise_exponential(20, 0.8)
    im4 = im.quantise_exponential(20, 0.9)

    fig, ax = plt.subplots(2, 4)
    plot_image(im1.image, ax=ax[0, 0])
    ax[0,0].set_title("q=17")
    ax[1,0].set_title(f"{im1.bits():.0f} bits")
    ax[1, 0].stairs(*im1.plot_histogram(False))

    plot_image(im2.image, ax=ax[0, 1])
    ax[0,1].set_title("k=20, r = 0.5")
    ax[1,1].set_title(f"{im2.bits():.0f} bits")
    ax[1, 1].stairs(*im2.plot_histogram(False))

    plot_image(im3.image, ax=ax[0, 2])
    ax[0,2].set_title("k=20, r = 0.8")
    ax[1,2].set_title(f"{im3.bits():.0f} bits")
    ax[1, 2].stairs(*im3.plot_histogram(False))
                    
    plot_image(im4.image, ax=ax[0, 3])
    ax[0,3].set_title("k=20, r = 0.9")
    ax[1,3].set_title(f"{im4.bits():.0f} bits")
    ax[1, 3].stairs(*im4.plot_histogram(False))
                                      # no padding around data
    fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.08)  # near-zero margins

    for axis in ax.flatten():
        axis.margins(0)
        # Minimal spines
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_linewidth(0.5)

        # Small ticks
        axis.tick_params(axis='both', labelsize=7, length=2, pad=2)
    
    plt.show()

def histogram_decreasing_spacing(image):
    im = Image(image)
    
    im1 = im.quantise(17)
    im2 = im.quantise_exponential(17, 1.1)
    im3 = im.quantise_exponential(17, 1.2)
    im4 = im.quantise_exponential(17, 1.3)

    print(im1.rms(im))
    print(im2.rms(im))
    print(im3.rms(im))
    print(im4.rms(im))

    fig, ax = plt.subplots(2, 4)
    plot_image(im1.image, ax=ax[0, 0])
    ax[0,0].set_title("q=17")
    ax[1,0].set_title(f"{im1.bits():.0f} bits")
    ax[1, 0].stairs(*im1.plot_histogram(False))

    plot_image(im2.image, ax=ax[0, 1])
    ax[0,1].set_title("k=17, r = 1.1")
    ax[1,1].set_title(f"{im2.bits():.0f} bits")
    ax[1, 1].stairs(*im2.plot_histogram(False))

    plot_image(im3.image, ax=ax[0, 2])
    ax[0,2].set_title("k=17, r = 1.2")
    ax[1,2].set_title(f"{im3.bits():.0f} bits")
    ax[1, 2].stairs(*im3.plot_histogram(False))
                    
    plot_image(im4.image, ax=ax[0, 3])
    ax[0,3].set_title("k=17, r = 1.3")
    ax[1,3].set_title(f"{im4.bits():.0f} bits")
    ax[1,3].stairs(*im4.plot_histogram(False))
                                      # no padding around data
    fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.08)  # near-zero margins

    for axis in ax.flatten():
        axis.margins(0)
        # Minimal spines
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_linewidth(0.5)

        # Small ticks
        axis.tick_params(axis='both', labelsize=7, length=2, pad=2)
    
    plt.show()


# histogram_increasing_spacing(bridge)
# histogram_decreasing_spacing(bridge)