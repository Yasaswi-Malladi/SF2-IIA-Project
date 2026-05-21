from cued_sf2_lab.laplacian_pyramid import rowdec, rowint, quantise, bpp
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.simple_image_filtering import halfcos, convse
import matplotlib.pyplot as plt
import numpy as np
import scipy

def conv_h(X, h):
    Y = np.copy(X)
    return convse(convse(Y.T, h).T, h)

def rms(X, t):
    # return pow(np.sum((X-t)**2.0)/(X.size),0.5)
    return np.std(X-t)

def pyramid(X, h, n):
    x = [np.copy(X)]
    p = []
    for i in range(0, n):
        x.append(rowdec(rowdec(x[i], h).T, h).T )
        p.append(x[i] - rowint(rowint(x[i+1], 2*h).T,2*h).T)
    p.append(x[-1])
    return p

def ipyramid(p, h):
    a = [x for x in p]
    for i in range(len(p)-1, 0, -1):
        a[i-1] = a[i-1] + rowint(rowint(a[i], 2*h).T,2*h).T
    return a[0]

def pyramid_quantise(p, l):
    a = [0]*len(p)
    for i in range(0, len(l)):
        a[i] = quantise(p[i], l[i])
    return a

def ipyramid_quantise(p, h, l):
    return ipyramid(pyramid_quantise(p, l), h)

def optimal_pyramid(p, h, t):
    def optimisation_func(l):
        quantised = ipyramid_quantise(p, h, l)
        return np.std(t-quantised)
    return np.clip(scipy.optimize.minimize(optimisation_func, [20]*len(p), method='CG').x, 0, 255)

# def optimal_pyramid_const_step(p, h, t):
#     def optimisation_func(l):
#         quantised = ipyramid_quantise(p, h, [l]*len(p))
#         return np.std(t-quantised)
#     return np.clip(scipy.optimize.minimize(optimisation_func, 5, method='CG').x, 0, 255)

def optimal_pyramid_const_step(p, h, X, target_rms):
    def error_diff(step):
        Z = ipyramid_quantise(p, h, [step] * len(p))
        return np.std(X - Z) - target_rms
        
    result = scipy.optimize.brentq(error_diff, 1, 255)
    return result


def bits_to_store(p):
    bpps = [bpp(x) for x in p]
    sizes = [x.size for x in p]
    bits = [bpps[i]*sizes[i] for i in range(0, len(p))]
    return np.sum(bits)

# def comp_ratio(ref, comp):
#     return bpp(ref) / bpp(comp)



X=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]- 128.0
p = pyramid(X, [.25, .5, .25], 4)

# fig, ax = plt.subplots(1, 5)
# plt.suptitle("Convolution with h = Gcos([n pi]/[N+1])")
# for i, n in enumerate([3, 11, 33, 55, 101]):
#     im = conv_h(X, halfcos(n))
#     E = np.sum(im**2)
#     plot_image(im, ax=ax[i])
#     ax[i].set_title(f"n={n} E={E:.0e}")
# plt.show()

# fig, ax = plt.subplots(1, 3)
# ax[0].set_title("Original quantised(17)")
# ax[1].set_title("1L quantised(17)")
# ax[2].set_title("4L quantised(17)")
# h =  [.25, .5, .25]
# p_1 = pyramid(X,h, 1)
# l_1 = [17, 17] #10.3
# p_4 = pyramid(X,h, 4)
# l_4 = [17, 17,17,17, 17]
# X_quantised = quantise(X, 17)
# p_1_quantised = pyramid_quantise(p_1, l_1)
# p_4_quantised = pyramid_quantise(p_4, l_4)
# print("sizes", bits_to_store([X_quantised]), bits_to_store(p_1_quantised), bits_to_store(p_4_quantised))
# print("rms", rms(X, X_quantised), rms(X, ipyramid(p_1_quantised, h)), rms(X, ipyramid(p_4_quantised, h)))
# print("bpps", [bpp(x) for x in p_1_quantised],  [bpp(x) for x in p_4_quantised])
# plot_image(X_quantised, ax=ax[0])
# plot_image(ipyramid(p_1_quantised, h), ax=ax[1])
# plot_image(ipyramid(p_4_quantised, h), ax=ax[2])
# plt.show()

# fig, ax = plt.subplots(1, 5)
# plot_image(p[0], ax=ax[0])
# plot_image(p[1], ax=ax[1])
# plot_image(p[2], ax=ax[2])
# plot_image(p[3], ax=ax[3])
# plot_image(p[4], ax=ax[4])
# plt.title("Convolution with h = [0.25, .5, .25]")
# ax[0].set_title("First highpass")
# ax[1].set_title("Second highpass")
# ax[2].set_title("Third highpass")
# ax[3].set_title("Fourth highpass")
# ax[4].set_title("Lowpass")
# plt.show()

# fig, ax = plt.subplots(1, 3)
# recovered = ipyramid(p, [.25, .5, .25])
# plot_image(recovered, ax = ax[1])
# plot_image(X, ax=ax[0])
# plot_image(ipyramid_quantise(pyramid(X, [.25, .5, .25], 2), [.25, .5, .25], [0, 100]), ax=ax[2])

h = [1/16, 4/16, 6/16, 4/16,  1/16]
# h = [0.25, 0.5, .25]
fig, ax = plt.subplots(1, 2)
X_quantised = quantise(X, 17)
p = pyramid(X,h, 4)
l = [optimal_pyramid_const_step(p, h,X, np.std(X-X_quantised))]*len(p)
p_recovered = ipyramid_quantise(p, h,l)
p_quantised = pyramid_quantise(p, l)
print(l)
plot_image(X_quantised, ax=ax[0])
# plot_image(ipyramid_quantise(pyramid(X, h, 4), h,l), ax=ax[1])
plot_image(p_recovered, ax=ax[1])
print("rms", rms( X, p_recovered),np.std(X-X_quantised) )
print(bits_to_store([X_quantised]))
print(bits_to_store(p_quantised))
ax[0].set_title(f"Quantised(17) rms:{rms(X, X_quantised):.2f}")
ax[1].set_title(f"4L@{l[0]:.2f} rms:{rms(X, p_recovered):.2f}")
plt.show()

# print(bits_to_store(quantise(X, 17)))
# print(bits_to_store(quantise(X, 30)))

# fig, ax = plt.subplots(1, 3)
# recovered = ipyramid(p, [.25, .5, .25])
# plot_image(recovered, ax = ax[1])
# plot_image(X, ax=ax[0])
# plot_image(ipyramid_quantise(pyramid(X, [.25, .5, .25], 2), [.25, .5, .25], [0, 100]), ax=ax[2])