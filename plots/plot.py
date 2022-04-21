
from os import listdir
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, Button
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






# creates a plot with sliders, one or two, using a callback function eg.:
"""
def callback(slider1, slider2, **params):
    x = np.linspace(0,10,1000)
    return np.sin(slider1*x) + np.cos(slider2*x) +c*x
"""
# and us the function with the callback, like this:
"""
c=2
plot_slider(callback, [[-2,0,2],[-2,0,2]], c=c)

"""
def plot_slider(callback, sliders, plot_type="plot", **callback_params):
    fig, ax = plt.subplots()
    if len(sliders) == 2:
        out = callback(sliders[0][1], sliders[1][1], **callback_params)
    else:
        out = callback(sliders[0][1], **callback_params)

    if plot_type == "plot":
        x = np.arange(0, len(out))
        plot, = ax.plot(x, out)
    elif plot_type == "image":
        plot = ax.imshow(out)
    plt.subplots_adjust(left=0.25, bottom=0.25)


    ax_slider1 = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_1 = Slider(
        ax=ax_slider1,
        label='Slider 1',
        valmin=sliders[0][0],
        valmax=sliders[0][2],
        valinit=sliders[0][1],
    )

    if len(sliders)==2: # define second slider
        ax_slider2 = plt.axes([0.1, 0.25, 0.0225, 0.63])
        slider_2 = Slider(
            ax=ax_slider2,
            label='Slider 2',
            valmin=sliders[1][0],
            valmax=sliders[1][2],
            valinit=sliders[1][1],
            orientation='vertical'
        )


    def update(val):
        if plot_type == "plot":
            if 'slider_2' in locals():
                plot.set_data(x, callback(slider_1.val, slider_2.val, **callback_params))
            else:
                plot.set_data(x, callback(slider_1.val, **callback_params))
        elif plot_type == "image":
            if 'slider_2' in locals():
                plot.set_data(callback(slider_1.val, slider_2.val, **callback_params))
            else:
                plot.set_data(callback(slider_1.val, **callback_params))
        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw_idle()

    slider_1.on_changed(update)
    if 'slider_2' in locals(): slider_2.on_changed(update)

    plt.show()