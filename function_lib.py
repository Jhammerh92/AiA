
from os import listdir
from  mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import scipy as sp
import scipy.signal as signal
import cv2


# Get lighting object for shading surface plots.
from matplotlib.colors import LightSource

# Get colormaps to use with lighting object.
from matplotlib import cm

# for MKR week 5
import maxflow
import maxflow.fastmin


### WEEK 1

def imread(im_str, as_type=False, load_type=np.float64, **kwargs):
    im = skimage.io.imread(im_str, kwargs)
    if as_type: im = im.astype(load_type)
    return im

def imshow(*ims, ax=None, cmap= "gray", im_type=np.uint8):
    
    if not (ax is None):
        ax.imshow(ims[0].astype(im_type), cmap=cmap) # can only do one image then
    else:
        # plt.figure()
        # plt.imshow(im, cmap=cmap)
        n = len(ims)
        if n > 1:
            r = int(np.floor(np.sqrt(n)))
            c = int(np.ceil(np.sqrt(n)))
            if r*c < n:
                r += 1
            fig, axes = plt.subplots(r,c)
            i = 0
            for ax in axes:
                if n == 2:
                    if i < n:
                        ax.imshow(ims[i].astype(im_type), cmap= cmap)
                    ax.axis("off")
                    i += 1 
                else:
                    for ax_ in ax:
                        if i < n:
                            ax_.imshow(ims[i].astype(im_type), cmap= cmap)
                        ax_.axis("off")
                        i += 1
        else:
            fig = plt.figure()
            plt.imshow(ims[0].astype(im_type), cmap=cmap)
            plt.axis("off")


## kernels
def std_or_t(std, t):
    if std is None and not(t is None):
        std = np.sqrt(t)
    elif std is None and t is None:
        std = 1
    return std

def gaussian_kernel(x=None, std=None, t=None, dim=0, diff=0):
    std = std_or_t(std, t)
        
    if x is None:
        s = 4 * std
        #x = np.arange(-s,s+1)
        x = np.linspace(-s, s, int(2*s+1 // 1)) # centers the range around 0
        y = x.T

    if diff == 1:
        if dim == 2:
            kernel = np.array(-x*np.exp(-x**2/(2*std**2))/(std**3*np.sqrt(2*np.pi)),ndmin=2)
        else:    
            kernel = np.array(-x*np.exp(-x**2/(2*std**2))/(std**3*np.sqrt(2*np.pi)),ndmin=2) # not correct!
    elif diff == 2:
        if dim == 2:
            xx,yy = np.meshgrid(x,y)
            kernel = np.array(  - 1/(np.pi*std**4) *(1 - (xx**2 + yy**2)/(2*std**2))* np.exp(-(xx**2+yy**2)/(2*std**2))  ,ndmin=2)
        else:
            kernel = np.array(- (std**2-x**2) * np.exp(- x**2/(2*std**2)) / (std**5 * np.sqrt(2*np.pi)),ndmin=2)
    elif diff == 3:
        if dim == 2:
            xx,yy = np.meshgrid(x,y)
            kernel = np.array(  - 1/(np.pi*std**4) *(1 - (xx**2 + yy**2)/(2*std**2))* np.exp(-(xx**2+yy**2)/(2*std**2))  ,ndmin=2)
        else:
            kernel = np.array(- (-3*std**2+x**2) * x* np.exp(-x**2/(2*std**2)) / (std**7 * np.sqrt(2*np.pi)),ndmin=2)
    elif diff == 4:
        if dim == 2:
            xx,yy = np.meshgrid(x,y)
            kernel = np.array(  - 1/(np.pi*std**4) *(1 - (xx**2 + yy**2)/(2*std**2))* np.exp(-(xx**2+yy**2)/(2*std**2))  ,ndmin=2)
        else:
            kernel = np.array(( 3*std**4 - 6* x**2 * std**2 + x**4) * np.exp(-x**2/(2*std**2)) / (std**9 * np.sqrt(2*np.pi)),ndmin=2)

    else:
        kernel = np.array(np.exp(-x**2/(2*std**2))/(std*np.sqrt(2*np.pi)),ndmin=2)
        if dim==2:
            kernel = kernel * kernel.T

    


    return kernel

def diff_kernel(s=1):
    kernel = np.array(np.arange(-s,s+1),ndmin=2)
    kernel = kernel/(2*s)
    return kernel


# convolution

def conv_image1D(im, kernel, dim=0): # virker nok ikke korrekt?
    im_out = np.copy(im)
    #kernel = np.flip(kernel)
    if dim == 1 or dim == 0:
        for i, y in enumerate(im_out):
            im_out[i,:] = np.convolve(y,kernel[0], mode='same')
    if dim == 2 or dim == 0:
        for i, x in enumerate(im_out.T):
            im_out[:,i] = np.convolve(x, kernel[0], mode='same')

    return im_out

def conv_image2D(im, kernel):
    return signal.convolve2d(im, kernel, mode='same', fillvalue=np.mean(im))


def image_derivative(im, std=None, t=None, dim=1):
    std = std_or_t(std, t)
    kernel = gaussian_kernel(std=std, diff=1, dim=1)
    if dim == 2:
        kernel = kernel.T
    im_deriv = conv_image2D(im, kernel)

    return im_deriv

def oriented_image_derivative(im, std=None, t=None, angle=np.pi/2):
    std = std_or_t(std, t)
    kernel_x = gaussian_kernel(std=std, diff=1, dim=1)
    kernel_y = kernel_x.T
    L_x = conv_image2D(im, kernel_x)
    L_y = conv_image2D(im, kernel_y)

    L_ang = L_x * np.cos(angle) + L_y * np.sin(angle)

    return L_ang





def segmentation(A,B=None):
    if B is None:
        B = A
    a = A[:,0:-2]
    b = B[:,1:-1]
    mask = (a != b)*1
    total = np.sum(mask)   
    return total, mask


def curve_smoothing(X, alpha=0.5, beta=0.5):
    I = np.eye(X.shape[0])
    a1 = np.roll(I,1, axis=1)
    a2 = np.roll(I,-1, axis=1)
    A = a1 + a2 + -2*I

    b1 = -1*np.roll(a1,1, axis=1)
    b2 = -1*np.roll(a2,-1, axis=1)
    B = b1 + b2 + 4*a1 + 4*a2 - 6*I

    smoothing_matrix = np.linalg.inv(I - alpha*A - beta*B)
    X_out = smoothing_matrix @ X

    return X_out, smoothing_matrix


def curve_smoothing_simple(X, lamb=0.5):
    I = np.eye(X.shape[0])
    l1 = np.roll(I,1, axis=1)
    l2 = np.roll(I,-1, axis=1)
    L = l1 + l2 + -2*I
    
    X_out = X
    
    X_out = np.linalg.inv(I - lamb*L)@X_out
    return X_out


def variation(X):
    a = X[:,0:-2]
    b = X[:,1:-1]
    V = np.sum(np.abs(a-b))
    return V


def image_unwrap(X, center=None, res_deg=1):
    #res_deg = round(res_deg)
    nsamples = int(360/res_deg)
    sizex,sizey = X.shape
    if center is None:
        center = np.array([round(sizex/2),round(sizey/2)])
    #print(center[0],center[1],sizex-center[0],sizey-center[1])
    radius = int(np.min([center[0],center[1],sizex-center[0],sizey-center[1]]))
    #print(radius)
    #rsamples = np.linspace(0,radius,300)
    U = np.empty([radius, nsamples])
    #print(U.shape)
    samples = np.linspace(0,360,nsamples)
    for i,ang in enumerate(samples):
        for r in range(radius):
            theta = np.deg2rad(ang)
            x = int(r * np.cos(theta) + center[0]) 
            y = int(r * np.sin(theta) + center[1])
            #print(r, i, x,y)
            U[r, i] = X[x,y]

    return U 



def load_slices_3D(folder):
    elements_in_folder = sorted(listdir(folder))
    nelements = len(elements_in_folder)
    size = skimage.io.imread(folder +"/" +elements_in_folder[0]).shape
    structure = np.empty((size[0],nelements,size[1]))
    for i, element in enumerate(elements_in_folder):
        slic = skimage.io.imread(folder + "/" +element)
        structure[:,i,:] = slic

    return structure


def create_surface_mesh(volume, surface_level=128, step_size=1 , spacing=(1,1,1)):
    verts, faces, normals, values = skimage.measure.marching_cubes(volume,
                                                       level=surface_level, 
                                                       spacing=spacing, 
                                                       gradient_direction='descent', 
                                                       step_size=step_size, 
                                                       allow_degenerate=True, 
                                                       method='lewiner', 
                                                       #mask=None
                                                       )

    mesh = Poly3DCollection(verts[faces])
    surf_object = {"mesh": mesh,
                   "verts": verts,
                   "faces": faces,
                   "normals": normals,
                   "values": values,
                   "spacing": spacing,
                   "volume": volume}
    return surf_object


def plot_surface_mesh(mesh,verts,faces,normals,values,spacing,volume, ax=None, alpha=1, color_RGB=(255,54,57), ls_az_alt = (225.0, 45.0), contrast=(0.15, 0.95)):

    ls = LightSource(*ls_az_alt)

    # First change - normals are per vertex, so I made it per face.
    normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])

    # Next this is more asthetic, but it prevents the shadows of the image being too dark. (linear interpolation to correct)
    min = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
    max = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
    diff = max-min
    newMin = contrast[0]#0.15
    newMax = contrast[1]#0.95
    newdiff = newMax-newMin

    # Using a constant color, put in desired RGB values here.
    colourRGB = np.array((color_RGB[0]/255.0, color_RGB[1]/255.0, color_RGB[2]/255.0, 1.0))

    # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
    rgbNew = np.array([colourRGB*(newMin + newdiff*((shade-min)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])

    # Apply color to face
    mesh.set_facecolor(rgbNew)
    mesh.set_edgecolor(rgbNew)
    mesh.set_alpha(alpha)
    mesh.set_linewidth(1)
    

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.view_init(23,133)
    ax.set_xlim(0, volume.shape[0]*spacing[0])
    ax.set_ylim(0, volume.shape[1]*spacing[1])
    ax.set_zlim(0, volume.shape[2]*spacing[2])



## WEEK 2
# added diff to gaussian_kernel 

def create_im_scalespace(im, t=1, lvls=3, ax='lin'):
    Ls = [im]
    L = im
    T = t
    
    for l in range(lvls-1):
        ksize = int(np.sqrt(T))*5*2+1
        L_temp = cv2.GaussianBlur(L, (ksize,ksize), np.sqrt(T))
        #g = gaussian_kernel(t=T)
        #L_temp = conv_image1D(L, g, dim=0) # gives weird artefact aroun the edge, maybe it has to do with lack of zero padding
        Ls.append(L_temp)
        T = T + t
        L = L_temp
    return Ls


def laplacian(im, std=None, t=None):

    g_diff2 = gaussian_kernel(std=std,t=t, diff=2, dim=2)
    #L_xx = conv_image2D(im, g_diff2)
    #L_yy = conv_image2D(im, g_diff2.T)
    #L = (L_xx + L_yy)
    #g_diff2 = g_diff2*g_diff2.T
    L = conv_image2D(im, g_diff2)
    if not (t is None):
        L *= t
    else:
        L *= std**2

    return L


def find_blob_centers(im_lap, threshold = None):
    threshold = abs(np.max(im_lap)*0.95)
    x,y = im_lap.shape
    centers = []

    for i in range(1, x-1):
        for j in range(1, y-1):
            section = im_lap[i-1:i+2,j-1:j+2]
            section_max = np.max(section)
            section_min = np.min(section)
            section_2nd_max = np.partition(section.flatten(),-2)[-2]
            section_2nd_min = np.partition(section.flatten(),1)[1]
            center = im_lap[i,j]

            if abs(im_lap[i,j]) > threshold:
                if (center == section_max or center == section_min):
                    if section_max > section_2nd_max or section_min < section_2nd_min:
                        centers.append([j,i])
                        #print(section,center,sep="\n")
                        #print(i,j,"\n")

    return np.array(centers)


# WEEK 3

def calc_image_descriptor(im, std=None, t=None, normalize=True):
    g = gaussian_kernel(std=std, t=t, dim=1, diff=0)
    g_x = gaussian_kernel(std=std,t=t, dim=1, diff=1)
    g_xx = gaussian_kernel(std=std,t=t, dim=1, diff=2)
    g_xxx = gaussian_kernel(std=std,t=t, dim=1, diff=3)
    g_xxxx = gaussian_kernel(std=std,t=t, dim=1, diff=4)

    
    L = conv_image1D(conv_image1D(im, g, dim=1), g, dim=2)
    
    L_x = conv_image1D(conv_image1D(im, g_x, dim=1), g, dim=2)
    L_y = conv_image1D(conv_image1D(im, g, dim=1), g_x, dim=2)
    L_xx = conv_image1D(conv_image1D(im, g_xx, dim=1), g, dim=2)
    L_xy = conv_image1D(conv_image1D(im, g_x, dim=1), g_x, dim=2)
    L_yy = conv_image1D(conv_image1D(im, g, dim=1), g_xx, dim=2)
    L_xxx = conv_image1D(conv_image1D(im, g_xxx, dim=1), g, dim=2)
    L_xxy = conv_image1D(conv_image1D(im, g_xx, dim=1), g_x, dim=2)
    L_xyy = conv_image1D(conv_image1D(im, g_x, dim=1), g_xx, dim=2)
    L_yyy = conv_image1D(conv_image1D(im, g, dim=1), g_xxx, dim=2)
    L_xxxx = conv_image1D(conv_image1D(im, g_xxxx, dim=1), g, dim=2)
    L_xxxy = conv_image1D(conv_image1D(im, g_xxx, dim=1), g_x, dim=2)
    L_xxyy = conv_image1D(conv_image1D(im, g_xx, dim=1), g_xx, dim=2)
    L_xyyy = conv_image1D(conv_image1D(im, g_x, dim=1), g_xxx, dim=2)
    L_yyyy = conv_image1D(conv_image1D(im, g, dim=1), g_xxxx, dim=2)


    desc = np.stack([L, L_x, L_y, L_xx, L_xy, L_yy,L_xxx,L_xxy,L_xyy,L_yyy,L_xxxx,L_xxxy,L_xxyy,L_xyyy,L_yyyy], axis = 2)
    if normalize: 
        desc -= np.mean(desc, axis=(0,1))
        desc /= np.std(desc, axis=(0,1))

    return desc


def multiscale_image_descriptor(im, scales_t=[1,2,4]):

    n = len(scales_t)
    F_multi = np.empty((im.shape[0],im.shape[1],15*n))
    #print(F_multi.shape)

    for i, t in enumerate(scales_t):
        F_multi[:,:, 15*(i): 15*(i+1)] = calc_image_descriptor(im,t=t)

    return F_multi


# WEEK 4









# WEEK 5

# one-clique potential function
def one_clique_potential(im, prior, mu, prior_is_labels=False):
    if prior_is_labels: prior = mu[prior] 
    V1 = (prior - im)**2
    U = np.sum(V1)
    return U, V1, prior

def minimize_one_clique(im,mu):
    n_mu = len(mu)+1
    prior_stack_ones = np.ones_like(im)
    prior_stack = np.ones_like(im)
    for n in range(2,n_mu):
        prior_stack = np.dstack((prior_stack, prior_stack_ones*n))
    prior_stack -= 1
    mus = mu[prior_stack.astype(np.int64)]
    im_stack = im.copy()
    if im.ndim == 1:
        im_stack = im_stack[None,:]
    im_stack = im_stack[:,:,None]
    im_stack = np.tile(im_stack,(1,1,n_mu-1))
    V1 = (mus - im_stack)**2
    edges = V1[0]
    min_configuration = np.argmin(V1, axis=2)
    V1 = np.min(V1, axis=2)
    U_min = np.sum(V1)
    return U_min, V1, min_configuration, edges


def two_clique_potential(im, labels, beta=100):
    V2_UD = 0
    
    if im.ndim >=2:
        neighbors_left = labels[:,:-1] # create all sets of up/down pairs..
        neighbors_right = labels[:,1:]
        neighbors_up = labels[:-1,:]
        neighbors_down = labels[1:,:]
        mask_up_down = neighbors_up == neighbors_down
        V2_UD = beta * ~mask_up_down

    
    neighbors_left = labels[:-1] # create all sets of left/right pairs
    neighbors_right = labels[1:]

    mask_left_right = neighbors_left == neighbors_right
    V2_LR = beta * ~mask_left_right
    U = np.sum(V2_LR) + np.sum(V2_UD)
    
    #print("C2 energy ", U)

    return U



def threshold_segmentation(im, peaks=None, thresholds=None): # OR peak segmentation
    im = im.copy().astype(np.float64)
    stack = None
    if not (peaks is None):
        stack = np.tile(np.copy(im[:,:,None]),(1,1,len(peaks)))
        stack = np.abs(stack - peaks.reshape((1,1,-1)) )
        # for i, peak in enumerate(peaks):
        #     stack[:,:,i] =  np.abs(stack[:,:,i] - peak) # subtract the peak values from each stack and find the index of the min value, which is then the segment index
        segments = np.argmin(stack, axis=2)

    elif not(thresholds is None):
        n_seg = len(thresholds)
        segments = np.ones_like(im) * n_seg # assign the last index to the entire image to skip the last segmentation check i.e. assume that if not assigned in the loop then they must fall into the last segment
        prev_mask = np.zeros_like(im).astype(np.bool8) # a mask of previously checked pixels
        for i, thr in enumerate(thresholds):
            mask = (im - thr) <= 0
            segments[mask & ~prev_mask] = i
            prev_mask = mask | prev_mask # lift the prev_mask

    return segments, stack


def posterior_potential(im,labels, mu, beta=100):
    labels = labels * 1
    if labels.ndim > im.ndim:
        labels = labels[0]

    U1 = one_clique_potential(im, labels, mu=mu, prior_is_labels=True)[0]
    U2 = two_clique_potential(im, labels, beta)
    return U1 + U2


"""Max FLow starts here"""
def graph_cut_binary_segmentation(im, tedges=None, beta=100):
    if tedges is None:
        tedges = im

    g = maxflow.GraphFloat(*im.shape)
    nodeids = g.add_grid_nodes(im.shape)
    g.add_grid_edges(nodeids, weights=beta, symmetric=True)

    # 1D case
    if im.ndim == 1:
        g.add_grid_tedges(nodeids, tedges[0,:], tedges[1,:])

    #2D case
    if im.ndim == 2:
        try:
            max_val = np.iinfo(im.dtype).max # if int type
        except:
            max_val = np.finfo(im.dtype).max # if float type, might be very large

        g.add_grid_tedges(nodeids, tedges, max_val - tedges) # 255 is assuming uint8 format!

    # run g.maxflow() to calculate
    g.maxflow()

    # get the segmentation of the nodes
    segm = (~ g.get_grid_segments(nodeids))*1 # "not" it, and make to ints

    return segm



# multilabel segmentation using alpha expansion algorithm
def graph_cut_multi_segmentation(im, mu, alpha=1 , beta=100):
    """ Usage found at https://gist.github.com/CrisDamian/2fa3fdd0074d3c7e3f98e02c124eda69 """
    # alpha wieghts the change from big differences in pixel values while beta weights just change
    labels, D = threshold_segmentation(im, peaks=mu) # the 3d stack is d
    #AiA.imshow(D/255, im_type=D.dtype)
    B = beta - beta * np.eye(len(mu))
    V = alpha * np.abs(mu.reshape((-1,1)) - mu.reshape((1,-1))) # 
    segm = maxflow.fastmin.aexpansion_grid(D, V+B, labels=labels )
    return segm



