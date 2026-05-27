from Image import *

X=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]- 128.0
im = Image(X)
reference = Image(quantise(X, 17))

# DCT optimisation
# im = Image(X)
# step, r = im.comp_ratio_dct(8, 17)
# print(step, r)
# im.DCT(8)
# im.quantise(step)
# im.iDCT(8)
# im.plot()

# pyramid optimisation
# im = Image(X)
# reference = im.quantise(17, False)
# h = [.25, .5, .25]
# step, r = im.comp_ratio_pyramid_mse(h, 2, 17)
# print(step, r)
# im.pyramid(h, 2)
# im.quantise_pyramid([step]*3)
# im.ipyramid(h)
# print(im.rms(X))
# print(reference.rms(X))
# im.plot()

# im = Image(X)
# h = [.25, .5, .25]
# im.pyramid(h, 4)
# im.quantise_pyramid([30, 30, 30, 30, 30])
# im.ipyramid(h)
# im.plot()

# im = Image(X)
# print(len(X))
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