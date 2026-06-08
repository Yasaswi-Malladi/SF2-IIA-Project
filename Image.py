from cued_sf2_lab.dct import *
from cued_sf2_lab.familiarisation import *
from cued_sf2_lab.laplacian_pyramid import *
from cued_sf2_lab.lbt import *
from cued_sf2_lab.simple_image_filtering import *
from cued_sf2_lab.dwt import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


class Image():
    image: np.ndarray
    BIT_BUDGET = 40960
    mean: int
    _pyramid: np.ndarray

    # General
    def __init__(self, image, make_zero_mean=False):
        im = np.array(image)
        self.image = im.copy()

        if make_zero_mean:
            self.mean = self.image.mean()
            self.image = self.image-self.mean
        else:
            self.mean = 0

    def get(self):
        return self.image +self.mean

    def _conv_h(self, h):
        Y = np.copy(self.image)
        return convse(convse(Y.T, h).T, h)           

    def rms(self, T):
        if type(T) is Image:
            return np.std(self.image- T.image)
        return np.std(self.image- T)
    
    def bits(self):
            return bpp(self.image)*self.image.size
    
    def quantise(self, step, rise=None):
        return Image(quantise(self.image, step, rise1=rise))

    def copy(self):
        return Image(self.image.copy()) 
    
    def quantise_exponential(self, k:int,r: int,  mode="linear"):
        i = np.arange(30)
        differences = np.round(4 + (k - 4) * r**i)       # tends to 4 as i -> inf
        levels = np.concatenate([-np.cumsum(differences)[::-1],[0], np.cumsum(differences)])   # s2[i+1] - s2[i] = s1[i]

        image = self.image
        diffs = np.abs(image[..., None] - levels)  
        indices = np.argmin(diffs, axis=-1)              # index of nearest level: (H, W, 3)
        quantised = levels[indices]
        return Image(quantised)


    def lzw_encode(self) -> list[int]:
        data = self.image.flatten()
        data = (data+self.mean*np.ones(len(data))).astype(np.uint8)
        # Initialize dictionary with all single values (0-255)
        dictionary = {(i,): i for i in range(256)}
        next_code = 256

        w = ()
        output = []

        for val in data:
            wc = w + (int(val),)
            if wc in dictionary:
                w = wc
            else:
                output.append(dictionary[w])
                dictionary[wc] = next_code
                next_code += 1
                w = (int(val),)

        if w:
            output.append(dictionary[w])

        return output

    def lzw_decode(codes: list[int], width=256, dtype=np.uint8) -> np.ndarray:
        # Initialize dictionary with single values
        dictionary = {i: (i,) for i in range(256)}
        next_code = 256

        result = []
        w = dictionary[codes[0]]
        result.extend(w)

        for code in codes[1:]:
            if code in dictionary:
                entry = dictionary[code]
            elif code == next_code:
                entry = w + (w[0],)  # Special case
            else:
                raise ValueError(f"Bad code: {code}")

            result.extend(entry)
            dictionary[next_code] = w + (entry[0],)
            next_code += 1
            w = entry

        return np.array(result, dtype=dtype).reshape(-1, width)

    # Plotting
    def plot_examples(images):
        fig, ax = plt.subplots(1, 5)
        for i, im in enumerate(images):
            plot_image(im, ax=ax[i])
    
    def plot(self):
        fig, ax = plt.subplots()
        plot_image(self.image, ax = ax)
        plt.show()

    def plot_histogram(self, plot = True):
        counts, bins = np.histogram(self.image, bins=256)
        if plot:
            plt.stairs(counts, bins)
            plt.show()
        return counts, bins
  
    # DWT
    def DWT(self, h1 = [-1/8, 2/8, 6/8, 2/8, -1/8], h2= [-1/4, 2/4, -1/4], l = 1):
        m = len(self.image)
        Y = dwt(self.image)
        for i in range(1, l):
            m = m//2
            Y[:m,:m] = dwt(Y[:m,:m], h1, h2)
        return DWT(Y)
    
    def iDWT(self, g1=[1/2, 1, 1/2], g2=[-1/4, -2/4, 6/4, -2/4, -1/4], l=1, apply=True):
        m = len(self.image) // pow(2, l-1)
        X = self.copy().image
        Y = idwt(X[:m, :m])
        X[:m, :m] = Y
        for i in range(l-2, -1, -1):
            m = len(self.image) // pow(2, i)
            Y = idwt(X[:m, :m], g1, g2)
            X[:m, :m] = Y
        return Image(X)

class DWT(Image):
    def MSE(h1=[-1/8, 2/8, 6/8, 2/8, -1/8], h2=[-1/4, 2/4, -1/4],g1 = [1/2, 1, 1/2], g2 = [-1/4, -2/4, 6/4, -2/4, -1/4],l=1, N=256):  
        e = np.full((3, l+1), np.inf)
        
        for level in range(l):
            m = N // pow(2, level+1)
            quadrant_centres = [
                (m//2,     m + m//2), 
                (m + m//2, m//2),
                (m + m//2, m + m//2),
            ]
            
            for k, (cy, cx) in enumerate(quadrant_centres):
                impulse = Image(np.zeros((N, N)))
                impulse = impulse.DWT(h1, h2, l=level+1)
                impulse.image[cy, cx] = 100
                impulse = impulse.iDWT(g1, g2, l=level+1)
                e[k, level] = np.sqrt(np.mean(impulse.image**2))
        
        m = N // pow(2, l)
        impulse = Image(np.zeros((N, N)))
        impulse = impulse.DWT(h1, h2, l=l)
        impulse.image[m//2, m//2] = 100
        impulse = impulse.iDWT(g1, g2, l=l)
        e[0, l] = np.sqrt(np.mean(impulse.image**2))
        
        e = e / e.min()
        return 1 / e

    def copy(self):
        return DWT(self.image.copy()) 
    
    def plot(self):
        fig, ax = plt.subplots()
        plot_image(self.image, ax = ax)
        plt.show()
    
    def quantise(self, M, l=1):
        im = self.copy().image
        for i in range(0, l):
            m = len(self.image) // pow(2, i+1)
            quadrants = [im[:m, m:], im[m:, :m], im[m:, m:]]
            for k in [0, 1, 2]:
                quadrants[k] = quantise(quadrants[k], M[k, i])

        m = len(self.image) // pow(2, l)
        im[:m, :m] = quantise(im[:m, :m], M[0, l])
        return DWT(im)

    def bits(self, l=1):
        def _split(arr, l):
            if l <= 1:
                return [half for row in np.vsplit(arr, 2) for half in np.hsplit(row, 2)]
            else:
                quadrants = [half for row in np.vsplit(arr, 2) for half in np.hsplit(row, 2)]
                squares = _split(quadrants[0], l-1)
                squares.extend(quadrants[1:])
                return squares
        Im = _split(self.image, l)
        bits = 0
        for Y in Im:
            bits += bpp(Y) * Y.size
        return bits

    def comp_ratio(image_array,mse = True, h1=[-1/8, 2/8, 6/8, 2/8, -1/8], h2=[-1/4, 2/4, -1/4],g1 = [1/2, 1, 1/2], g2 = [-1/4, -2/4, 6/4, -2/4, -1/4],l=1, step=17):
        reference = Image(quantise(image_array, step))
        reference_rms = reference.rms(image_array)
        
        def _dwt_optimisation_func(step):
            Z = Image(image_array)
            Z = Z.DWT(h1, h2, l)
            Z = Z.quantise(step_ratio*step, l)
            Z = Z.iDWT(g1, g2, l)
            return abs(Z.rms(image_array) - reference_rms)
        
        if mse:
            step_ratio = np.array(DWT.MSE(h1, h2, g1, g2, l=l))
        else:
            step_ratio = np.ones((3, l+1))
        
        opt_step = minimize_scalar(_dwt_optimisation_func, bracket=(0, 255),  method='golden', tol=0.001).x
        bits_ref = reference.bits()
        Z = Image(image_array)
        Z = Z.DWT(h1, h2, l)
        M = step_ratio*opt_step
        Z = Z.quantise(step_ratio*opt_step, l)
        bits_scheme = Z.bits(l=l)
        comp_ratio = bits_ref / bits_scheme
        return (M, comp_ratio)
    
lighthouse=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]-128.0
bridge=load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})[0]-128.0

# im = Image(lighthouse)
# print(im.quantise(17).bits())
# print(im.quantise(17).rms(im))

# exp_quantised = im.quantise_exponential(r = 0.2, k= 25)
# print(exp_quantised.bits())
# print(exp_quantised.rms(im))
# exp_quantised.plot()
# exp_quantised.plot_histogram()
