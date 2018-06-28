import numpy as np
from skimage import filters,img_as_float
from scipy import ndimage as nd

def nuclei_filtering(img_stack):
    """
    This function remove the background from the nuclei 
    For the sigma I seleced a value quite bigger than
    the nuclei size in order to remove them from the 
    image. I used what showed on the gaussian filter code page and on this
    link on stackoverflow: 
    http://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy

    Arguments
    -----------

    img_stack: np.array float64
        3D numpy array with the image
    
    Returns
    -----------

    filtered_image: np.array float64 
        2D flattened image 

    """

    # Clean the image from the background
    img_stack = img_stack-filters.gaussian(img_stack,sigma=(2,100,100))
    
    # Remove the negative values        
    img_stack[img_stack<0] = 0

    # Flatten the image
    filtered_image = np.amax(img_stack,axis=0)

    return filtered_image


def smFISH_filtering(img_stack):
    """
    This function remove the background from the smFISH and enhance the dots.
    
    Arguments
    -----------

    img_stack: np.array float64
        3D numpy array with the image
    
    Returns
    -----------

    filtered_image: np.array float64 
        2D flattened image 
    """

    # Use a gaussian with kernel bigger than the dots to estimate the background
    # and remove it from the image
    img_stack = img_stack-filters.gaussian(img_stack,sigma=(1,8,8))
    img_stack[img_stack<0] = 0

    # Enhance the dots
    img_stack = nd.gaussian_laplace(img_stack,sigma=(0.2,0.5,0.5))

    img_stack = -img_stack # the peaks are negative so invert the signal
    img_stack[img_stack<0] = 0 # All negative values set to zero 

    # Flatten the image
    filtered_image = np.amax(img_stack,axis=0)

    return filtered_image
