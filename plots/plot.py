
from os import listdir
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d
import numpy as np
import skimage.io
import scipy as sp
import scipy.signal as signal
import cv2

# Get lighting object for shading surface plots.
from matplotlib.colors import LightSource

# Get colormaps to use with lighting object.
from matplotlib import cm


def simple_slice(arr, inds, axis):
        # this does the same as np.take() except only supports simple slicing, not
        # advanced indexing, and thus is much faster
        sl = [slice(None)] * arr.ndim
        sl[axis] = inds
        return arr[tuple(sl)]


def plot_slices(arr, dim=0):
    
    n = arr.shape[dim]

    fig, axes = plt.subplots(int(np.floor(np.sqrt(n)))+1,int(np.ceil(np.sqrt(n))))
    i = 0
    for ax in axes:
        for ax_ in ax:
            if i < n:
                ax_.imshow(simple_slice(arr,i,dim), cmap= "gray")
            ax_.axis("off")
            i += 1

def plot_images(ims):
    n = len(ims)

    if n > 1:
        fig, axes = plt.subplots(int(np.floor(np.sqrt(n)))+1,int(np.ceil(np.sqrt(n))))
        i = 0
        for ax in axes:
                for ax_ in ax:
                    if i < n:
                        ax_.imshow(ims[i], cmap= "gray")
                    ax_.axis("off")
                    i += 1
    else:
        fig = plt.figure()
        plt.imshow(ims[0], cmap= "gray")
        plt.axis("off")


def plot_im_as_surf(im, ax=None, **kwargs):
    x,y = im.shape

    X, Y = np.meshgrid(np.arange(x),np.arange(y))

    if ax is None:
        fig, ax = plt.subplots(1)
        
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, im, **kwargs)