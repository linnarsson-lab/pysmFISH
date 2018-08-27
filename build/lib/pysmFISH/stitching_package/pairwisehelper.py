"""Helper functions for the class pairwise alignment
Creates its own logger object when imported.

Functions:

"""

import numpy as np
import skimage.transform as smtf
import logging
# import matplotlib.pyplot as plt

from . import inout

logger = logging.getLogger(__name__)


####################Third level helper functions####################
def get_overlapping_region(tile_1, tile_2,
                           overlap_ind_x, overlap_ind_y, direction):
    """Given the overlap indexes calculate the overlap in direction.

    Parameters:
    -----------

    tile_1: np.array
        2D or 3D numpy array representing an image.
    tile_2: np.array
        2D or 3D numpy array representing an image.
    overlap_ind_x: int
        The difference in pixels between the tile corners in the x/column direction.
    overlap_ind_y: int
        The difference in pixels between the tile corners in the y/row direction.
    direction: str
        Valid values: 'left' or 'top'. Denotes if
        the two tiles neighbour on the right and left or
        at the bottom and top.

    Returns:
    --------

    overlap_1: np.array
        2D or 3D numpy arrays. The parts of each image that overlap.
    overlap_2: np.array
        2D or 3D numpy arrays. The parts of each image that overlap.

    """
    if (direction == 'left' and tile_1.ndim == 2):
        # Adjust y overlap on left:
        if overlap_ind_y == 0:
            overlap_1 = tile_1[:, -overlap_ind_x:]
            overlap_2 = tile_2[:, :overlap_ind_x]
        elif overlap_ind_y < 0:
            # For negative difference in y index select overlap:
            overlap_1 = tile_1[-overlap_ind_y:, -overlap_ind_x:]
            overlap_2 = tile_2[:overlap_ind_y, :overlap_ind_x]
        else:
            # For positive y index select overlap
            overlap_1 = tile_1[:-overlap_ind_y, -overlap_ind_x:]
            overlap_2 = tile_2[overlap_ind_y:, :overlap_ind_x]
    elif (direction == 'left' and tile_1.ndim == 3):
        # Adjust y overlap on left:
        if overlap_ind_y == 0:
            overlap_1 = tile_1[:, :, -overlap_ind_x:]
            overlap_2 = tile_2[:, :, :overlap_ind_x]
        elif overlap_ind_y < 0:
            # For negative difference in y index select overlap:
            overlap_1 = tile_1[:, -overlap_ind_y:, -overlap_ind_x:]
            overlap_2 = tile_2[:, :overlap_ind_y, :overlap_ind_x]
        else:
            # For positive y index select overlap
            overlap_1 = tile_1[:, :-overlap_ind_y, -overlap_ind_x:]
            overlap_2 = tile_2[:, overlap_ind_y:, :overlap_ind_x]
    elif (direction == 'top' and tile_1.ndim == 2):
        # Adjust x overlap:
        if overlap_ind_x == 0:
            overlap_1 = tile_1[-overlap_ind_y:, :]
            overlap_2 = tile_2[:overlap_ind_y, :]
        elif overlap_ind_x < 0:
            # For negative difference in x index select overlap:
            overlap_1 = tile_1[-overlap_ind_y:, -overlap_ind_x:]
            overlap_2 = tile_2[:overlap_ind_y, :overlap_ind_x]
        else:
            # For positive x index select overlap
            overlap_1 = tile_1[-overlap_ind_y:, :-overlap_ind_x]
            overlap_2 = tile_2[:overlap_ind_y, overlap_ind_x:]
    elif (direction == 'top' and tile_1.ndim == 3):
        # Adjust x overlap:
        if overlap_ind_x == 0:
            overlap_1 = tile_1[:, -overlap_ind_y:, :]
            overlap_2 = tile_2[:, :overlap_ind_y, :]
        elif overlap_ind_x < 0:
            # For negative difference in x index select overlap:
            overlap_1 = tile_1[:, -overlap_ind_y:, -overlap_ind_x:]
            overlap_2 = tile_2[:, :overlap_ind_y, :overlap_ind_x]
        else:
            # For positive x index select overlap
            overlap_1 = tile_1[:, -overlap_ind_y:, :-overlap_ind_x]
            overlap_2 = tile_2[:, :overlap_ind_y, overlap_ind_x:]
    else:
        logger.warning("The overlapping region could not be retrieved\n"
                       + "The overlap direction and dimension of the image do not match any known case\n"
                       + "Direction: {} Dimension: {}".format(direction,
                                                              tile_1.ndim))

    return overlap_1, overlap_2


def calculate_PCM(pic_a, pic_b):
    """Calculate the phase correlation matrix of pic_a and pic_b.

    Parameters:
    -----------

    pic_a: np.array
        First picture
    pic_b: np.array
        Second picture

    Returns:
    --------

    r: np.array
        Normalized PCM matrix

    Notes:
    ------
        pic_a and pic_b should have the same size


    """
    # From: http://stackoverflow.com/questions/2771021/is-there-an-image-phase-correlation-library-available-for-python
    # And: https://github.com/michaelting/Phase_Correlation/blob/master/phase_corr.py
    # Calculate phase correlation matrix
    # Calculate nD fourier transform of first image
    G_a = np.fft.fftn(pic_a)
    # Calculate nD fourier transform of second image
    G_b = np.fft.fftn(pic_b)
    conj_b = np.ma.conjugate(G_b)  # Take the complex conjunggate of G_b
    R_raw = G_a * conj_b  # Multiply G_a by the complex conjugate of G_b
    R = R_raw / np.absolute(R_raw)  # Normalize elementwise
    # Inverse fourier transform to get correlation, and use only the real part
    r = np.fft.ifftn(R).real
    # inout.plot_3D(r)
    #plt.figure()
    #plt.imshow(r, 'gray', interpolation = 'none')
    #plt.show()

    return r


def calculate_PCM_method2(pic_a, pic_b):
    """Calculate the phase correlation matrix of pic_a and pic_b.

    Method 2 differs from calculate_PCM in that it calculates the PCM
    per, layer in the image therefore this function is only applicable
    to 3D images/complete z-stacks.
    In fact it is only used when in pairwisesingle.py if method ==
    'calculate per layer'.

    Parameters:
    -----------

    pic_a: np.array
        First picture
    pic_b: np.array
        Second picture

    Returns:
    --------

    r: np.array
        Normalized PCM matrix

    Notes:
    ------
        pic_a and pic_b should have the same size
    """
    r       = np.empty(pic_a.shape)
    for i in range(len(pic_a[:, 0, 0])):
        r[i, :, :] = calculate_PCM(pic_a[i, :, :], pic_b[i, :, :])
    return r


def calc_translated_pics(trans, overlap1, overlap2,
                         round_size=False):
    """Calculate wich part of 2 pictures overlap after translation.

    Translates overlap2 and returns the parts of overlap1 and
    overlap2 that overlap after translation. Parts that do not
    overlap are cropped. The alternative would be to fill the parts
    that do not overlap with zeros, but this seems to mess up the
    covariance calculation more than cropping the pictures.
    Here, one problem is that we may crop unnecessarily much
    because the cropping is done on the overlap and not on the whole
    tile.

    Parameters:
    -----------

    trans: np.array
        1 by 2 np-array with y and x translation.
    overlap1: np.array
        Image
    overlap2: np.array
        Image that is supposed to overlap with overlap1
    round_size: bool
        If True the final image sizes will be
        rounded to the nearest integer before warping the
        images. (Default: False)

    Returns:
    --------

    shifted_a: np.array
        Overlap a shifted and cropped to the same size as shifted_b
    shifted_b: np.array
        Overlap b shifted and cropped to the same size as shifted_a
    """
    # Pass to calc_translated_pics_3D if there are more then 2 dimensions
    if overlap1.ndim > 2:
        return calc_translated_pics_3D(trans, overlap1, overlap2,
                                       round_size=round_size)

    # Empty transformation matrix
    trans_matrix = np.eye(3)

    # Size after shift
    y_size, x_size = overlap2.shape
    # Always make new size smaller
    y_size -= abs(trans[0])
    x_size -= abs(trans[1])

    if round_size:
        y_size = int(round(y_size))
        x_size = int(round(x_size))

    # Transform a with transform to cut  on left and top
    if (trans[0] < 0):
        trans_matrix[1, 2] = abs(trans[0])
    if (trans[1] < 0):
        trans_matrix[0, 2] = abs(trans[1])
    # Or transform a with empty matrix (0,0) to make smaller on right
    logger.debug("x and y size: {} {}".format(x_size, y_size))
    shifted_a = smtf.warp(overlap1, trans_matrix,
                          output_shape = (y_size, x_size))
    # Reset transformation matrix
    trans_matrix = np.eye(3)

    # Set y transformation
    if trans[0] > 0:
        trans_matrix[1, 2] = trans[0]
    # Set x transformation
    if trans[1] > 0:
        trans_matrix[0, 2] = trans[1]

    # Transform b with empty matrix (0,0) to give it the right size
    shifted_b = smtf.warp(overlap2, trans_matrix,
                          output_shape=(y_size, x_size))

    return shifted_a, shifted_b


def calc_translated_pics_3D(trans, overlap1, overlap2,
                            round_size=False):
    """Calculate wich part of two 3D pictures overlap after translation.

    Translates overlap2 and returns the parts of overlap1 and
    overlap2 that overlap after translation.
    This function works in 3D but does not translate in z.
    Parts that do not overlap are cropped. The alternative would be to
    fill the parts that do not overlap with zeros, but this seems to
    mess up the covariance calculation more than cropping the pictures.
    Here, one problem is that we may crop unnecessarily much
    because the cropping is done on the overlap and not on the whole
    tile.

    Parameters:
    -----------

    Parameters:
    -----------

    trans: np.array
        1 by 2 np-array with y and x translation.
    overlap1: np.array
        Image
    overlap2: np.array
        Image that is supposed to overlap with overlap1
    round_size: bool
        If True the final image sizes will be
        rounded to the nearest integer before warping the
        images. (Default: False)

    Returns:
    --------

    shifted_a: np.array
        Overlap a shifted and cropped to the same size as shifted_b
    shifted_b: np.array
        Overlap b shifted and cropped to the same size as shifted_a
    """

    logger.debug("Trans: {}".format(trans))
    # Size after shift
    z_size, y_size, x_size = overlap2.shape
    # Always make new size smaller
    y_size -= abs(trans[1])
    x_size -= abs(trans[2])

    if round_size:
        y_size = int(round(y_size))
        x_size = int(round(x_size))

    # Do x and y translation for each layer:
    # Empty transformation matrix
    trans_matrix = np.eye(3)

    # Transform a with transform to cut  on left and top
    if (trans[1] < 0):
        # y trans
        trans_matrix[1, 2] = abs(trans[1])
    if (trans[2] < 0):
        # x trans
        trans_matrix[0, 2] = abs(trans[2])
    # Or transform a with empty matrix (0,0) to make smaller on right
    logger.debug("Trans, z, x and y size: {} {} {} {}"
                 .format(trans, z_size, x_size, y_size))
    logger.debug("Matrix: {}"
                 .format(trans_matrix))
    shifted_a = np.empty((z_size, y_size, x_size))
    logger.debug("Shape overlap 1, after z transform: {}"
                 .format(overlap1.shape))
    for i in range(overlap1.shape[0]):
        shifted_a[i, :, :] = smtf.warp(overlap1[i, :, :], trans_matrix,
                                       output_shape=(y_size, x_size))
        # plt.figure()
        # plt.imshow(shifted_a[i,:,:])
        # plt.show()

    # Do x and y transform in each layer
    # Reset transformation matrix
    trans_matrix = np.eye(3)

    # Set y transformation
    if trans[1] > 0:
        trans_matrix[1, 2] = trans[1]
    # Set x transformation
    if trans[2] > 0:
        trans_matrix[0, 2] = trans[2]

    shifted_b = np.empty((z_size, y_size, x_size))
    # Transform b with empty matrix (0,0) to give it the right size
    for i in range(overlap2.shape[0]):
        shifted_b[i, :, :] = smtf.warp(overlap2[i, :, :], trans_matrix,
                                       output_shape=(y_size, x_size))
    return shifted_a, shifted_b


def xcov_nd(pic_1, pic_2):
    """Calculate the normalized cross covariance of pic_1 and pic_2.

    Parameters:
    -----------

    pic_1: np.array
        Numpy array representing a gray value picture, same size as pic_2
    pic_2: np.array 
        Numpy array representing a gray value picture, same size as pic_1

    Returns:
    --------

    : float
         The normalized crosscovariance of pic_1 and pic_2 (coVar / (stDev1 * stDev2))
    monocolor: bool   
        True if the one or both of the pictures contained only one color.
    """
    # Converted from Java code used in Preibisch plugin:
    # https://github.com/tomka/imglib/blob/master/mpicbg/imglib/algorithm/fft/PhaseCorrelation.java
    # Alternative use: scipy.signal.correlate2d
    monocolor = False
    avg1 = np.mean(pic_1)
    avg2 = np.mean(pic_2)
    dist1M = pic_1 - avg1
    dist2M = pic_2 - avg2

    # Calculate var and co variance for the whole picture, per pixel, elementwise multiplication
    coVarMatrix = dist1M * dist2M
    var1Matrix = dist1M * dist1M
    var2Matrix = dist2M * dist2M

    # Take the average
    coVar   = np.mean(coVarMatrix)
    var1    = np.mean(var1Matrix)
    var2    = np.mean(var2Matrix)

    stDev1 = np.sqrt(var1)
    stDev2 = np.sqrt(var2)

    # All pixels had the same color....
    if (stDev1 == 0 or stDev2 == 0):
        monocolor = True
        if (stDev1 == stDev2 and avg1 == avg2):
            return (1, monocolor)
        else:
            return (0, monocolor)

    # Compute correlation coeffienct
    return (coVar / (stDev1 * stDev2), monocolor)


def display_overlap(overlap1, overlap2, best_trans, best_cov,
                    plot_order):
    """Plot two pictures and show how they overlap.

    This function will only plot when allow_plot is True and in inout
    matplotlib is imported and plot_available is True.

    Parameters:
    -----------

    overlap1: np.array
        2d np-array representing an image
    overlap2: np.array
        2d np-array representing an image
    best_trans: np.array
        1 by 2 np array with an y and x value for the best transition
    best_cov: float
        Number indicating the goodness of the overlap between the two pictures
    plot_order: list
        The order in wich subplots should be placed if when plotting overlap1 
        and 2 and overlap rgb
    """
    allow_plot = False
    if allow_plot:
        logger.debug("Displaying overlap")
        logger.debug("best trans, cov: {}, {}"
                     .format(best_trans, best_cov))

        shifted_a, shifted_b = calc_translated_pics(best_trans,
                                                    overlap1, overlap2,
                                                    round_size=True)
        best_cov = xcov_nd(shifted_a, shifted_b)

        # overlap_rgb         = np.zeros((shifted_a.shape[0], shifted_a.shape[1],  3))
        # overlap_rgb[:,:,0]  = smex.rescale_intensity(shifted_a)
        # overlap_rgb[:,:,1]  = smex.rescale_intensity(shifted_b)

        title_text = "Pairwise overlap, cov: {}".format(best_cov)

        # inout.display_tiles([shifted_a,
        #                         shifted_b,
        #                         overlap_rgb],
        #                         plot_order, fig_nr = "overlap check",
        #                         maximize = False, main_title = title_text,
        #                         block = True)
        overlap_sum = shifted_a + shifted_b
        inout.display_tiles([shifted_a,
                             shifted_b,
                             overlap_sum],
                            plot_order, fig_nr="overlap check",
                            maximize=False, main_title=title_text,
                            block=True)
        # if shifted_a.ndim == 3:
        #    inout.plot_3D(overlap_sum)
    else:
        return None
