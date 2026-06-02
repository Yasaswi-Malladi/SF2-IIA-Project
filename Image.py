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
    _pyramid: np.ndarray

    # General
    def __init__(self, image):
        im = np.array(image)
        self.image = im.copy()

    def _conv_h(self, h):
        Y = np.copy(self.image)
        return convse(convse(Y.T, h).T, h)           

    def rms(self, T):
        if type(T) is Image:
            return np.std(self.image- T.image)
        return np.std(self.image- T)
    
    def bits(self):
            return bpp(self.image)*self.image.size
    
    def quantise(self, step, apply = True):
        if apply:
            self.image = quantise(self.image, step)
        else:
            return Image(quantise(self.image, step))

    def copy(self):
        return Image(self.image.copy()) 
    
    # Plotting
    def plot_examples(images):
        fig, ax = plt.subplots(1, 5)
        for i, im in enumerate(images):
            plot_image(im, ax=ax[i])
    
    def plot(self):
        fig, ax = plt.subplots()
        plot_image(self.image, ax = ax)
        plt.show()
    
    # Pyramid
    def pyramid(self, h, n, apply=True):
        x = [np.copy(self.image)]
        p = []
        for i in range(0, n):
            x.append(rowdec(rowdec(x[i], h).T, h).T )
            p.append(x[i] - rowint(rowint(x[i+1], 2*h).T,2*h).T)
        p.append(x[-1])
        p = [x for x in p]
        if apply:
            self._pyramid = p
        else:
            return p

    def ipyramid(self, h, apply=True):
        a = [x for x in self._pyramid]
        for i in range(len(self._pyramid)-1, 0, -1):
            a[i-1] = a[i-1] + rowint(rowint(a[i], 2*h).T,2*h).T
        
        if apply:
            self.image = a[0]
        else:
            return a[0]
        
    def MSE_pyramid(h, n, N=256):
        z = Image(np.zeros((N, N)))
        z.pyramid(h, n)
        e = []
        for i in range(0, n+1):
            impulse = Pyramid(z._pyramid).copy()
            layer = impulse._pyramid[i]
            H, W = layer.shape
            layer[H//2, W//2] = 100
            e.append(z.rms(impulse.ipyramid(h, False))) #rms -> sqrt mse
        
        e = [1/x for x in e]
        e = [x/e[0] for x in e]
        return e
    
    def bits_pyramid(self):
        bits = [bpp(x)*x.size for x in self._pyramid]
        return np.sum(bits)    

    def comp_ratio_pyramid(self, h, n, step):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        p = self.pyramid(h, n, False)
        def _pyramid_optimisation_func(step):
            Z = Pyramid(p)
            Z.quantise_pyramid([step]*(n+1))
            Z.ipyramid(h)
            return abs(Z.rms(self.image) - reference_rms)
            
        opt_step = minimize_scalar(_pyramid_optimisation_func, bounds=(0, 255), method='bounded').x
        bits_ref = reference.bits()
        Z = Pyramid(p)
        Z.quantise_pyramid([opt_step]*(n+1))
        bits_scheme = Z.bits_pyramid()
        comp_ratio = bits_ref / bits_scheme
        return (opt_step, comp_ratio)

    def comp_ratio_pyramid_mse(self, h, n, step):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        p = self.pyramid(h, n, False)
        mse_ratio = np.array(Image.MSE_pyramid(h, n, len(self.image)))
        def _pyramid_optimisation_func(step):
            Z = Pyramid(p)
            Z.quantise_pyramid(mse_ratio*step)
            Z.ipyramid(h)
            return abs(Z.rms(self.image) - reference_rms)
            
        opt_step = minimize_scalar(_pyramid_optimisation_func, bounds=(0, 255), method='bounded').x
        bits_ref = reference.bits()
        Z = Pyramid(p)
        Z.quantise_pyramid([opt_step]*(n+1))
        bits_scheme = Z.bits_pyramid()
        comp_ratio = bits_ref / bits_scheme
        return (opt_step, comp_ratio)

    def quantise_pyramid(self, l, apply = True):
        a = [0]*len(self._pyramid)
        for i in range(0, len(l)):
            a[i] = quantise(self._pyramid[i], l[i])
        if apply:
            self._pyramid = a
        else:
            return a  

    # DCT
    def DCT(self, N, apply=True):
        c = dct_ii(N)
        if apply:
            self.image = colxfm(colxfm(self.image,c).T,c).T
        else:
            return Image(colxfm(colxfm(self.image,c).T,c).T)

    def iDCT(self, N, apply=True):
        c = dct_ii(N)
        if apply:
            self.image = colxfm(colxfm(self.image.T,c.T).T,c.T)
        else:
            return Image(colxfm(colxfm(self.image.T,c.T).T,c.T))
        
    def bits_dct(self, N):
        w = len(self.image) // N
        bits = 0
        for y in range(0, w*N, w):
            for x in range(0, w*N, w):
                Y = self.image[x: x+w, y: y+w]
                bits += bpp(Y) * Y.size
        return bits
        # def recursive
    
    def regroup_N(self, N):
        return regroup(self.image, N)/N

    def regroup(self, N, apply = True):
        if apply:
            self.image = regroup(self.image, N)
        else:
            return Image(regroup(self.image, N))
    
    def comp_ratio_dct(self, N: int, step :float):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        
        def _DCT_optimisation_func(step):
            return abs(reference_rms - self.DCT(N, False).quantise(step, False).iDCT(N, False).rms(self.image))
        
        opt_step = minimize_scalar(_DCT_optimisation_func, bounds=(0, 255), method='bounded').x
        bits_ref = reference.bits()
        bits_scheme = self.DCT(N, False).quantise(opt_step, False).regroup(N, False).bits_dct(N)
        comp_ratio = bits_ref / bits_scheme

        return (opt_step, comp_ratio)

    # POT / LBT
    def POT(self, N,s=(1 + (5 ** 0.5)) / 2, apply=True):
        Pf = pot_ii(N, s)[0]
        t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
        Xp = self.image.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        c = dct_ii(N)
        if apply:
            self.image = colxfm(colxfm(Xp,c).T,c).T
        else:
            return Image(colxfm(colxfm(Xp,c).T,c).T)
        
    def iPOT(self, N,s=(1 + (5 ** 0.5)) / 2, apply=True):
        Pr = pot_ii(N, s)[1]
        t = np.s_[N//2:-N//2] 
        c = dct_ii(N)
        Zp = colxfm(colxfm(self.image.T,c.T).T,c.T) #iDCT
        Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], Pr.T)
        if apply:
            self.image = Zp
        else:
            return Image(Zp)
    
    def comp_ratio_lbt(self,N:int,s:float,  step: float):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        
        def _POT_optimisation_func(step):
            return abs(reference_rms - self.POT(N,s, False).quantise(step, False).iPOT(N,s, False).rms(self.image))
        
        opt_step = minimize_scalar(_POT_optimisation_func, bounds=(0, 255), method='bounded').x
        bits_ref = reference.bits()
        bits_scheme = self.POT(N,s, False).quantise(opt_step, False).regroup(N, False).bits_dct(N)
        comp_ratio = bits_ref / bits_scheme

        return (opt_step, comp_ratio)

    # DWT
    def DWT_example(self, h1 = [-1/8, 2/8, 6/8, 2/8, -1/8], h2= [-1/4, 2/4, -1/4], apply=True):
        U = rowdec(self.image, h1)
        V = rowdec2(self.image, h2)
        UU = rowdec(U.T, h1).T
        UV = rowdec2(U.T, h2).T
        VU = rowdec(V.T, h1).T
        VV = rowdec2(V.T, h2).T
        image = np.block([[UU, UV], [VU, VV]])
        if apply:
            self.image = image
        else:
            return Image(image)
        
    def iDWT_example(self, g1=[1/2, 1, 1/2], g2=[-1/4, -2/4, 6/4, -2/4, -1/4], apply=True):
        top, bottom = np.vsplit(self.image, 2)
        UU, UV = (arr.copy() for arr in np.hsplit(top, 2))
        VU, VV = (arr.copy() for arr in np.hsplit(bottom, 2))
        Ur = rowint(UU.T, g1).T + rowint2(UV.T, g2).T
        Vr = rowint(VU.T, g1).T + rowint2(VV.T, g2).T
        Xr = rowint(Ur,g1) + rowint2(Vr,g2)
        if apply:
            self.image = Xr
        else:
            return Image(Xr)
    
    def DWT(self, h1 = [-1/8, 2/8, 6/8, 2/8, -1/8], h2= [-1/4, 2/4, -1/4], l = 1, apply=True):
        m = len(self.image)
        Y = dwt(self.image)
        for i in range(1, l):
            m = m//2
            Y[:m,:m] = dwt(Y[:m,:m], h1, h2)
        if apply:
            self.image = Y
        else:
            return Image(Y)
    
    def iDWT(self, g1=[1/2, 1, 1/2], g2=[-1/4, -2/4, 6/4, -2/4, -1/4], l=1, apply=True):
        m = len(self.image) // pow(2, l-1)
        X = self.copy().image
        Y = idwt(X[:m, :m])
        X[:m, :m] = Y
        for i in range(l-2, -1, -1):
            m = len(self.image) // pow(2, i)
            Y = idwt(X[:m, :m], g1, g2)
            X[:m, :m] = Y
        if apply:
            self.image = X
        else:
            return Image(X)
        
    def MSE_dwt(h1=[-1/8, 2/8, 6/8, 2/8, -1/8], h2=[-1/4, 2/4, -1/4],g1 = [1/2, 1, 1/2], g2 = [-1/4, -2/4, 6/4, -2/4, -1/4],l=1, N=256):   
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
                impulse.DWT(h1, h2, l=level+1)
                impulse.image[cy, cx] = 100
                impulse.iDWT(g1, g2, l=level+1)
                e[k, level] = np.sqrt(np.mean(impulse.image**2))
        m = N // pow(2, l)
        impulse = Image(np.zeros((N, N)))
        impulse.DWT(h1, h2, l=l)
        impulse.image[m//2, m//2] = 100
        impulse.iDWT(g1, g2, l=l)
        e[0, l] = np.sqrt(np.mean(impulse.image**2))
        
        e = e / e.min()
        return 1 / e
    
    def quantise_dwt(self, M, l=1, apply=True):
        im = self.copy().image
        for i in range(0, l):
            m = len(self.image) // pow(2, i+1)
            quadrants = [im[:m, m:], im[m:, :m], im[m:, m:]]
            for k in [0, 1, 2]:
                quadrants[k] = quantise(quadrants[k], M[k, i])

        m = len(self.image) // pow(2, l)
        im[:m, :m] = quantise(im[:m, :m], M[0, l])
        if apply:
            self.image = im
        else:
            return Image(im)

    def bits_dwt(self, l=1):
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

    def comp_ratio_dwt(self,h1=[-1/8, 2/8, 6/8, 2/8, -1/8], h2=[-1/4, 2/4, -1/4],g1 = [1/2, 1, 1/2], g2 = [-1/4, -2/4, 6/4, -2/4, -1/4],l=1, step=17):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        # print(f"reference {reference_rms}")
        
        def _dwt_optimisation_func(step):
            Z = Image(self.image)
            Z.DWT(h1, h2, l)
            Z.quantise_dwt(np.ones((3, l+1))*step, l)
            Z.iDWT(g1, g2, l)
            # Z.plot()
            # print(step, abs(Z.rms(self.image) - reference_rms))
            # print(Z.rms(self.image))
            return abs(Z.rms(self.image) - reference_rms)
            
        # opt_step = minimize_scalar(_dwt_optimisation_func, bounds=(0, 255), method='bounded', ).x
        opt_step = minimize_scalar(_dwt_optimisation_func, bracket=(0, 255),  method='golden', tol=0.001).x
        # print(f"optimal step: {opt_step}")
        bits_ref = reference.bits()
        Z = Image(self.image)
        Z.DWT(h1, h2, l)
        Z.quantise_dwt(np.ones((3, l+1))*opt_step, l)
        bits_scheme = Z.bits_dwt(l=l)
        # Z.iDWT(g1, g2, l)
        # print("inside rms", Z.rms(self.image))
        comp_ratio = bits_ref / bits_scheme
        return (opt_step, comp_ratio)
    
    def comp_ratio_dwt_mse(self,h1=[-1/8, 2/8, 6/8, 2/8, -1/8], h2=[-1/4, 2/4, -1/4],g1 = [1/2, 1, 1/2], g2 = [-1/4, -2/4, 6/4, -2/4, -1/4],l=1, step=17):
        reference = Image(quantise(self.image, step))
        reference_rms = reference.rms(self.image)
        mse_ratio = np.array(Image.MSE_dwt(h1, h2, g1, g2, l=l))
        
        def _dwt_optimisation_func(step):
            Z = Image(self.image)
            Z.DWT(h1, h2, l)
            Z.quantise_dwt(mse_ratio*step, l)
            Z.iDWT(g1, g2, l)
            return abs(Z.rms(self.image) - reference_rms)
        
        opt_step = minimize_scalar(_dwt_optimisation_func, bracket=(0, 255),  method='golden', tol=0.001).x
        bits_ref = reference.bits()
        Z = Image(self.image)
        Z.DWT(h1, h2, l)
        M = mse_ratio*opt_step
        Z.quantise_dwt(M, l)
        bits_scheme = Z.bits_dwt(l=l)
        comp_ratio = bits_ref / bits_scheme
        return (M, comp_ratio)

class Pyramid(Image):
    def __init__(self, pyramid):
        self._pyramid = pyramid
    
    def copy(self):
        p = [im.copy() for im in self._pyramid]
        return Pyramid(p)

# lighthouse=load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]- 128.0
# im = Image(lighthouse)
# print(im.bits())
# im.DWT(l=2)
# print(im.bits_dwt(2))


# dct 16x16 rms matched CR: 2.92/ 2.93
# dwt 4 levels const. step 6.67