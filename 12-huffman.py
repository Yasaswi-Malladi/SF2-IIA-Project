#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import IPython.display

from cued_sf2_lab.familiarisation import load_mat_img, plot_image


# # 12. Combined run-length / Huffman coding methods
# 
# <div class="alert alert-warning alert-block">
#     
# This notebook is incomplete!</div>
# 
# Up to this point, we have been using entropy as a measure of the number of bits for the
# compressed image.  Now we attempt to produce a vector of compressed image data
# which accurately represents the compression that can be achieved in practise.
# 
# Huffman codes are relatively
# efficient at coding data with non-uniform probability distributions, provided
# that the probability of any single event does not exceed 50%.  However, when
# an image is transformed by any of the energy compaction methods
# considered so far, a high proportion of the quantised coefficients are zero,
# so this event usually does have a probability much greater than 50%.  In fact
# it is only when this _is_ a high probability event that high compression
# can be achieved!  Therefore new ways of using Huffman codes have been
# developed to deal with this situation as efficiently as possible.

# ## 12.1 Baseline JPEG coding techniques
# 
# The standard Huffman coding solution, used by the baseline JPEG
# specification and most other image compression standards, is to code each
# non-zero coefficient combined with the number of zero coefficients which
# preceed it as a single event.
# 
# For example the sequence of coefficients:
# $$
# 3, 0, 0, -2, 0, 0, 0, 1, 0, -3, -1, 0, 0, 0, 0, 1, \ldots
# $$
# would be coded as the following 6 events:
# ```
# 0 zeros, 3
# 2 zeros, -2
# 3 zeros, 1
# 1 zero, -3
# 0 zeros, -1
# 4 zeros, 1
# ```
# 
# Each event has a certain probability (usually well below 50%) and
# can be coded efficiently with a standard Huffman code.  As
# formulated above, the number of combinations of amplitude and
# run-length can be very large, leading to a highly complex code.
# JPEG limits this complexity to only 162 combinations by
# restricting the maximum run-length to 15 zeros and by coding only
# the base-2 logarithm of the amplitude in the Huffman code, rounded
# up to integers from 1 to 10.  The sign bit and the remaining
# amplitude bits are then appended to the Huffman code word.  16 run lengths (0 to
# 15) and 10 log amplitudes (1 to 10) give 160 of the code words.
# The other two codewords are the end-of-block word (EOB),
# signifying no more non-zero coefficients in the current block, and
# the run-of-16 word (ZRL), which may be used repetitively ahead of
# another codeword for runs of 16 or more zeros.
# 
# JPEG is based on $8 \times 8$ DCT transformations of the image, and the data
# from each $8 \times 8$ block of DCT coefficients is coded as a block of
# Huffman codewords.  First the dc coefficient (top left corner) is coded.
# There is little penalty in using a fixed-length binary code for this, although
# JPEG uses differential and Huffman coding for slightly improved performance.
# Then the remaining 63 ac coefficients are arranged into a linear vector, by
# scanning the $8 \times 8$ block in a zig-zag manner corresponding to
# progressively increasing frequencies (see the JPEG standard, section 3, fig 5).  This places
# the larger low-frequency coefficients close together near the start of the
# vector (with short run lengths) and the smaller high-frequency coefficients
# spread out towards the end of the vector (with long run lengths).  The
# end-of-block word efficiently terminates the coding of each block after the
# last non-zero coefficient.
# 
# For further details of the JPEG techniques, referred to above, see the JPEG
# standard, sections 3.3 and 3.6 and appendices A.3, C, F.1.1, F.1.2, F.2.1,
# F.2.2, K.1, K.2, and K.3.  Note that for this project we ignore the higher
# layers of the JPEG specification, and do not align code segments with byte
# boundaries or use two-byte marker codes to identify different data segments.
# JPEG also permits arithmetic codes to be used instead of Huffman codes, but
# these are more complicated so we recommend that you should use the latter.
# 

# ## 12.2 Python implementation of Huffman coding
# 
# <div class="alert alert-warning alert-block">
#     
# This section is **very** incomplete!</div>
# 
# 
# The file at `cued_sf2_lab/jpeg.py` contains implementations of the following functions:
# 
# * `jpegenc`: perform simplified JPEG encoding of an image `X` into a matrix of variable length codewords `vlc`.
# * `jpegdec`: perform simplified JPEG decoding of a codeword matrix `vlc` into an image `Z`.
# * `quant1`: quantise a matrix into integers representing the quantiser step numbers, which is the form necessary to allow Huffman coding.
# * `quant2`: reconstruct a matrix from integers.  Together with `quant1` this is equivalent to `quantise`.
# * `runampl`: convert a vector of coefficients `a` into a matrix of run-length, log-amplitude and signed-remainder values `rsa`.
# * `huffenc`: convert a run/amplitude matrix `rsa` into a matrix of variable-length codewords `vlc`.
# * `huffdflt`: generate the specification table `hufftab`, for the default JPEG Huffman code tables for AC luminance
# or AC chrominance coefficients (JPEG specification, appendix K.3.3.2).
# * `huffdes`: design the specification table, `hufftab`, for optimised JPEG Huffman code tables using a histogram of
# codeword usage `huffhist`.
# * `huffgen`: generate the Huffman code tables, `huffcode` and `ehuf`, from `hufftab`.

# In[2]:


from cued_sf2_lab.jpeg import (
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen)


# In order to allow relatively fast decoding in Python, we have cheated a
# little in the format of the coded data.  Each variable-length codeword is
# stored as an integer element of the required word length in the first column
# of a 2-column matrix `vlc` and the length of the codeword in bits is
# stored next to it in the second column.  We do not bother to pack this data
# into a serial bit stream since it is awkward and time consuming to unpack in
# Matlab, and we have not got around to changing this now that the lab is in Python!
# The length of the bit-stream if it were packed can easily be obtained
# from `vlc[:,1].sum()`.

# To perform the simplified JPEG encoding, based on the $8 \times 8$ DCT,
# load the image in `X` and type:  `vlc, hufftab = jpegenc(X, qstep)`.
# You can inspect the codewords with `dict(zip(hufftab.huffval, hufftab.codes))`.

# In[3]:


# your code here


# This produces variable-length coded data in `vlc`, using quantisation step sizes of qstep.  To decode `vlc`, type:  `Z = jpegdec(vlc, qstep)`

# In[5]:


# your code here


# This `jpeg.py` file is given to you as examples of how to achieve a complete compression
# system. They have other options and outputs, and in general you will need to copy them to a new
# file and modify them to perform your own algorithms.
# 
# In `jpegenc`, there are two ways to specify the Huffman
# tables: either the default JPEG AC luminance or chrominance tables
# may be used; or custom tables may be designed, based on statistics
# in the histogram vector `huffhist`. To generate a valid
# histogram for `huffdes`, coding must be performed at least
# once using `huffdflt` instead, so `jpegenc` is written
# such that the default tables are used first and then, if required,
# the code is redesigned using custom tables. Note that if it is
# planned to use `huffdes` to generate an optimised Huffman code
# for each new image to be coded, then the specification tables `hufftab.bits` and `hufftab.huffval` must be sent with the compressed image,
# which costs (16 + 162) bytes = 1424 bits.  You should consider
# whether or not this is a sensible strategy.

# ## Going beyond JPEG and the DCT
# 
# If you have chosen the DCT as one of your energy compaction
# methods then it is fairly straightforward to follow the JPEG
# guidelines for coding the coefficients.  However if you have
# chosen one of the other methods then a modified scanning strategy is required.
# 
# It has already been mentioned that the LBT (which is at the heart of the JPEG-XR standard) is often coded several sub-blocks at a time. We can make a smaller LBT ($4 \times 4$ is the default) look like a $16 \times 16$ DCT by using the `regroup(Yq, 4)` function within each $16 \times 16$ block of `Yq`. The functions `jpegenc` and `jpegdec` have already been written to do this if the `M` argument (which specifies the {\em coding} block size) is larger than the `N` argument (which specifies the DCT block size).
# 
# The DWT (which is the basis of the JPEG2000 standard) can also be re-arranged to make it look similar to a DCT. For instance, a 3-level DWT could be re-arranged into an $8 \times 8$ block $B$ using coefficients from the same
# square spatial area:
# 
# 
# > 4 values from level 3: $B3$ = [$UU_3$ $VU_3$; $UV_3$, $VV_3$]  
# > 3 surrounding $2 \times 2$ blocks from level 2: $B2$ = [$B3$ $VU_2$; $UV_2$ $VV_2$]  
# > 3 surrounding $4 \times 4$ blocks from level 1: $B$ = [$B2$ $VU_1$; $UV_1$ $VV_1$]
# 
# It is not possible to achieve this sort of grouping using the simple `regroup` function, so we have provided a more complicated function 
# `dwtgroup(X,n)` which converts an n-level DWT sub-image set into blocks of size $N \times N$ (where $N = 2^n$) with the above type of grouping.  Try this
# function on some small regular matrices (e.g. `np.arange(16*16).reshape(16, 16)`) to
# see how it works. Note that `dwtgroup(X, -n)` reverses this grouping.

# In[7]:


from cued_sf2_lab.jpeg import dwtgroup
# this is a nice trick to get an array of coordinates; use `display(x)` to check what this is doing
i, j = np.indices((8, 8))
x = np.rec.array((i, j))
print(dwtgroup(x, 2))


# 
# With these modified scanning strategies, the JPEG run-length / log amplitude coding can then be used for each vector in the same way as for the DCT coefficients. However, these scanning strategies are not optimal, and do not represent those outlined in the JPEG2000 and JPEG-XR standards.
# 
# You should write versions of `jpegenc` and `jpegdec` for
# your chosen compression strategies and check the following:
# 1. The rms error (standard deviation) between the decoded and original
# images should be the same as for the equivalent quantisation
# strategies that were tested in the previous section on
# centre-clipped linear quantisers.  No extra errors should be
# introduced by the scanning or Huffman encode / decode operations.
# 
# 2. The number of bits required to code an image should be comparable with
# the value predicted from the entropy of the quantised coefficients (i.e.
# within about 20%).  Note that it is possible to code with fewer bits than
# predicted by the entropy because the run-length coding can take advantage of
# clustering of non-zero coefficients, which is not taken account of in the
# first-order entropy calculations.
# 
# 

# 
