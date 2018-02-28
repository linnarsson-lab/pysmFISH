import numpy as np
import skimage.util as smutil
import skimage.exposure as smex
from skimage import img_as_float, img_as_uint
import glob


# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# plt_available = True

plt_available = False
import logging
import warnings

from . import inout
from . import pairwisehelper as ph


"""Calculates the global corners, given the translations
for each tile. Also has functions to blend and stitch individual
images together in one large image, using the global corners.
"""

logger = logging.getLogger(__name__)


#################Functions to calculate global corners##############
def calc_corners_coord(tiles, transforms, md, nr_pixels, z_count):
    """Calculate global corner for each tile using coordinates.

    Calculates global corner for each tile using transforms and the
    microscope coordinates.

    Parameters:
    -----------

    tiles: list
        List of hdf5-references, references
        to the images in tile_file, with a
        reference for each tile. Here only used to check
        the length. If a tile does not have an
        associated image, its reference is None.
    transforms: np.array
        2d np array of size "number of tiles" by 2
        representing the y and x transform for each tile.
    md: object
        MicroscopeData object containing the tile set
        (to know the positioning of the tiles) and the y and
        x coordinates for each tile as documented by the
        microscope.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of the z
        axis). Should be 1 or None if the tile is 2D.

    Returns:
    --------
    : dict
        Dictionary containing keys corner_list and
        final_image_shape. Corner_list is a list of list, each
        list is a pair of an image number (int) and it's
        coordinates (numpy array containing floats).
        Final_image_shape is a tuple of size 2 or 3 depending on
        the number of dimensions and contains ints.
    """
    #Initialize the logger and arrays:
    logger.info("Filling corner list.")
    max_final_img       = np.zeros((2), dtype = float)
    min_final_img       = np.zeros((2), dtype = float)
    tile_inds           = range(len(tiles))
    temp_corner_list    = []

    #Get the masked array, used for checking for missing tiles
    masked_array    = np.ma.getmaskarray(md.tile_set.flat[:])
    logger.debug("masked_array: {}".format(masked_array))

    #Loop over the tiles in square order:
    for tile_ind in tile_inds:
        #Find out which image number belongs to the current tiles
        ind_coord       = md.tile_set.flat[:][tile_ind]
        #Check if tile is missing
        tile_missing    = masked_array[tile_ind]
        if not(tile_missing):
            logger.info("Calculating corner for tile index: {}  with "
                        + "tile number: {}"
                        .format(ind_coord, tile_ind))
            #Get the coordinates of the corner and adjust according
            #to the transform in transforms
            cur_corner = np.rint(np.array([md.y_coords[ind_coord],
                                           md.x_coords[ind_coord]])
                                 - transforms[tile_ind, -2:])
            logger.debug("Current transform, full: {} truncated: {}"
                         .format(transforms[tile_ind],
                                 transforms[tile_ind, -2:]))

            #Adjust the maximum size of the final image if necessary
            max_final_img   = np.maximum(max_final_img, cur_corner)
            min_final_img   = np.minimum(min_final_img, cur_corner)
            #Append the calculated corner to the temporary corner
            #list
            temp_corner_list.append(cur_corner)
        else:
            #If the tile we want to place is missing, the corner
            #will be flagged with np.nan values and added to the list
            logger.info("Tile is missing for tile index: {}  with "
                        + "tile number: {}"
                        .format(ind_coord, tile_ind))
            cur_corner = [np.nan, np.nan]
            temp_corner_list.append(cur_corner)
    #Adjust the temporary corner list according to the minimum of
    #the final image.
    #This way we get a new origin at min_final_image.
    logger.debug("min_final_img: {}".format(min_final_img))
    temp_corner_list   -= min_final_img
    #Place the temporary corner list in the global variable
    corner_list        = [[tile_inds[i], temp_corner_list[i]]
                                    for i in range(len(tile_inds))]
    #Adjust the shape of the final image
    if (z_count is None) or z_count == 1:
        final_image_shape  = tuple((max_final_img - min_final_img
                                    + nr_pixels).astype(int))
    else:
        final_image_shape  = tuple(np.append([z_count],
                                max_final_img - min_final_img
                                + nr_pixels).astype(int))
    #Give some feedback
    logger.info("Corners calculated")
    logger.debug("Corner_list: {}".format(corner_list))
    logger.info("Final image shape: {}".format(final_image_shape))

    return {'corner_list': corner_list,
            'final_image_shape': final_image_shape}


def apply_IJ_corners(tiles, corners, micData, nr_pixels):
    """Use corners from ImageJ for each tile to detemine corners.

    Parameters:
    -----------

    tiles: list
        List of hdf5-references, references
        to the images in tile_file, with a
        reference for each tile. Here only used to check
        the length. If a tile does not have an
        associated image, its reference is None.
    corners: list
        List of list, each list is a pair of an image number
        (int) and it's coordinates (numpy array containing
        floats).
    micData: object
        MicData object. Used to make an image list: the
        numbers of the images to be stitches sorted
        according to the tile indexes.
    nr_pixels: int
        Denoting size of the tile.

    Returns:
    --------
    : dict
        Same as calc_corners_coord() returns.
        Dictionary containing keys corner_list and
        final_image_shape. Corner_list is a list of list, each
        list is a pair of an image number (int) and it's
        coordinates (numpy array containing floats).
        Final_image_shape is a tuple of size 2 or 3 depending on
        the numer of dimensions and contains ints.
    """
    #Initialize the logger and arrays:
    logger.info("Filling corner list, using ImageJ")
    corner_list         = []
    max_final_img       = np.zeros((2))
    min_final_img       = np.zeros((2))
    tile_inds           = range(len(tiles))
    # Make a list of image numbers, matching with the numbers in the
    # image files
    flat_tile_set = micData.tile_set.flat[:]
    image_list = [micData.tile_nr[ind] if ind >= 0
                                    else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    #Loop over the tiles in square order:
    for tile_ind in tile_inds:
        #Find out which image number belongs to the current tiles
        image_nr        = image_list[tile_ind]
        #Find the corner that matches the image number
        cur_corner = next(item[1] for item in corners
                            if item[0] == image_nr)
        logger.info("Index found (coord, image_nr, tile): {}, {}, {}"
                        .format(cur_corner, image_nr, tile_ind))
        #Adjust the maximum size of the final image if necessary
        max_final_img   = np.maximum(max_final_img, cur_corner)
        min_final_img   = np.minimum(min_final_img, cur_corner)
        corner_list.append([tile_ind, cur_corner])

    #TODO: Adjust corner to start at (0,0)
    logger.debug("min_final_img: {}".format(min_final_img))
    final_image_shape  = tuple((max_final_img - min_final_img
                                    + nr_pixels).astype(int))
    logger.info("Corners determined")
    logger.debug("Corner_list: {}".format(corner_list))
    logger.info("Final image shape: {}".format(final_image_shape))

    return {'corner_list': corner_list,
        'final_image_shape': final_image_shape}


########### Functions to join the tiles into to final image ############
def make_mask(joining, nr_pixels, blending_mask):
    """Calculate the mask that indicates where tiles overlap.

    This mask will have the same size a the final image.
    This function assigns a float (1.0, 2.0, 3.0 or 4.0) to each pixel,
    indicating if 1, 2, 3 or 4 tiles are going to overlap in this pixel.

    Parameters:
    -----------

    joining: dict
        Dictionary containing the corner list
        (with key: 'corner_list') with the tile indexes
        and their corresponding corners
    nr_pixels: int      
        Indicates the size of the tiles
    blending_mask: pointer
        Dataset in an hdf5 file containing a 2D numpy
        array. Array has the size of final image.
    """
    logger.info("Making blending mask")
    for i, corner in joining['corner_list']:
        if not(np.isnan(corner[0])):
            #Get the right part of mask
            cur_mask = blending_mask[int(corner[0]):int(corner[0]) + int(nr_pixels),
                            int(corner[1]):int(corner[1]) + int(nr_pixels)]
            # Place it back, after plus one
            blending_mask[int(corner[0]):int(corner[0]) + int(nr_pixels),
                            int(corner[1]):int(corner[1]) + int(nr_pixels)] = \
                             cur_mask + np.ones(cur_mask.shape, dtype=np.float64)
            #print(i,corner)
            #print('intervals:',int(corner[0]),int(corner[0]) + #int(nr_pixels),int(corner[1]),int(corner[1]) + int(nr_pixels))


def generate_blended_tile(temp_file, im_file, tiles, tile_file, corner_ind_coord, nr_pixels, tile_set,
                blend, linear_blending, ubyte,
                nr_dim = 2):
    """Blend the tile if necessary and then save it temp_file.

    Parameters:
    -----------

    temp_file: pointer
        Pointer to hdf5 file withv following groups:
        tiles, temp_masks, ubytes.
        Each group contains as many datasets as there are
        tiles, the datasets are named after the the tile
        index found in the first element of corner.
        This function places a blended tile and a corner in
        data set that matches the tile ind argument.
    im_file: pointer
            Pointer to hdf5 file with dataset "blending_mask"
            which contains a numpy array. blending_mask should
            be 1 where ther is no overlap and 2, 3 or 4 where
            the respective number of tiles overlap.
            Other datasets in this file are: final_image
            and temp_mask
    tiles: list
        List of hdf5-references, references
        to the images in tile_file, with a
        reference for each tile. If a tile does not have
        an associated image, its reference is None.
    tile_file: pointer
        hdf5 file object. The opened file containing the tiles to stitch.
    corner_ind_coord: list
        Contains two elements, the first one is an
        int representing the tile index, the second one is
        a numpy array containing the corner's coordinates.
    nr_pixels: int
        Denoting size of the tile.
    tile_set: np.array
        Masked numpy array. The shape of the array
        indicates the shape of the tile set.
    blend: bool
        When True blending will be applied,
        when false no blending at all will be applied.
    linear_blending: bool
        When True blending will be linear
        and when False, blending will be non-linear.
    ubyte: bool
        Ubyte image will be saved when True. Only full resolution image will be saved when False.
    nr_dim: int
        If 3, the code will assume three dimensional
        data for the tile, where z is the first dimension
        and y and x the second and third. For any other
        value 2-dimensional data is assumed. (default: 2)
    """
    # Get corner
    i = corner_ind_coord[0]
    corner = corner_ind_coord[1]
    # Load tile
    if nr_dim == 3:
        cur_tile = inout.load_tile_3D(tiles[i], tile_file)
    else:
        cur_tile = inout.load_tile(tiles[i], tile_file)
    # Check if tile is empty
    if cur_tile.size:
        logger.info("\nBlending tile: {}".format(i))
        logger.info("Corner position in final image: {}"
                    .format(corner))
        #Pick the right region of the image with corner
        ymin = int(corner[0])
        ymax = int(corner[0]) + nr_pixels
        xmin = int(corner[1])
        xmax = int(corner[1]) + nr_pixels

        logger.debug(" Tile {}, Cur region: {} {} {} {}"
                            .format(i, ymin, ymax, xmin, xmax))
        if blend:
            #Get current mask and region to paste the tile into
            cur_mask   = im_file['blending_mask'][ymin:ymax, xmin:xmax]

            #Blend
            blended_tile, cur_temp_mask  = perform_blending_par_proof(i,
                    cur_mask, cur_tile,
                    linear_blending, tiles, tile_set, nr_pixels)

            #Save the blended tile and the adjusted mask (mask is just saved for debugging)
            logger.debug('shape tile in temp file: {}'
                        .format(temp_file['blended_tiles'][str(i)].shape))
            logger.debug('shape current tile: {}'
                        .format(blended_tile.shape))
            temp_file['blended_tiles'][str(i)][:] = blended_tile
            temp_file['temp_masks'][str(i)][:]  = cur_temp_mask

            #Place the low resolution region back,
            if ubyte:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',
                                    category = UserWarning)
                    temp_file['ubytes'][str(i)][:] = \
                                smutil.img_as_ubyte(blended_tile)
            logger.info("Blended tile {} placed into temp_group."
                        .format(i))
        else:
            # Place the region back without blending
            temp_file['blended_tiles'][str(i)][:] = cur_tile

            #Place the low resolution region back
            if ubyte:
                temp_file['ubytes'][str(i)][:] = \
                                smutil.img_as_ubyte(cur_tile)
            logger.info("Tile {} placed into temp_group without "
                        + "blending.".format(i))
    else:
        logger.info("Skipped empty tile, tile: {}".format(i))


def non_linear_blending(x):
    """Define sigmoid for non-linear blending

    The steepness and half value of the curve are hardcoded here and
    good for a 10% overlap. For other overlaps flexibity in the
    steepness may be good to implement. (Halfpoint should be the
    same)

    Parameters:
    -----------

    x: np.array
        1d np-array, used as x values in the sigmoid curve

    Returns:
    --------

    y : np.array
        1d np-array, y values corresponding to x after applying
        sigmoid function on them
    """
    y = 1 / (1 + np.exp((-20 * (x - 0.5))))

    #logger.debug("Non linear blending x {}".format(x))
    #logger.debug("Non linear blending y {}".format(y))
    #plt.figure("alpha")
    #plt.plot(x,y,'-*')
    #plt.show()
    return y


def non_linear_blending_corner(x,y):
    """Calculate blending weights for pixels where four tiles overlap

    Parameters:
    -----------

    x: np.array
        1d numpy array, x and y should be of the same length. The
        distance from the corner in the x direction for each pixel.
    y: np.array
        1d numpy array, x and y should be of the same length. The
        distance from the corner in the x direction for each pixel.

    Returns:
    --------
    : np.array
        1d numpy array, same size as x and y. Weight for each pixel.
    """
    return x * y


def perform_blending_par_proof(tile_ind, cur_mask,
                        cur_tile, linear_blending,
                        tiles, tile_set, nr_pixels):
    """Blend a tile with the background around it.

    Each pixel in the overlap gets a weight depending on its distance
    from the border.

    Parameters:
    -----------

    tile_ind: int
        Index of the current tile
    cur_mask: np.array
        The mask denoting were in the
        picture tile have already been placed. A value of
        0 means no picture, a value of 1 means that a
        picture has been placed, a value of 2 means that
        there is overlap with the current tile.
    cur_tile: np.array
        The warped image to be placed..
    linear_blending: bool
        If true perform linear blending,
        otherwise non-linear blending.
    tiles: list
        List of hdf5-references, references
        to the images in tile_file, with a
        reference for each tile. Here only used to check
        if a tile exists, if a tile does not have an
        associated image, its reference is None.
    tile_set: np.array
        np-array representing the shape of the tile set
    nr_pixels: int
        Denoting size of the tile.

    Returns:
    --------

    cur_tile: np.array
        The blended cur_tile, same type and
        size as the input argument cur_type.
    cur_temp_mask: np.array
        Array like the cur_mask as it was
        passed to the function, but with the pixels that
        overlap replaced by their respective weights. This
        mask can be plotted for debugging.
    """

    logger.info("Performing blending on tile {}".format(tile_ind))
    # Find overlapping indexes in the blending mask.
    overlapping_inds = np.where(cur_mask > 1.0)
    logger.debug("overlapping inds: {}".format(overlapping_inds))
    logger.debug("size of cur_mask: {}".format(cur_mask.shape))

    if overlapping_inds[0].any() or overlapping_inds[1].any():

        cur_corner_tl   = np.array([0,0])
        cur_corner_br   = np.array([nr_pixels,nr_pixels])

        # Make lists for the 4 borders, with large numbers for borders that should not be used:
        # Order: [top, left, bottom, right]
        border_direct   = np.ones(4, dtype = int) * 9999
        border_indirect = np.ones(4, dtype = int) * 9999
        true_int = 0            # Value to use if overlap at a border exists
        unr_tile_ind = np.unravel_index(tile_ind, tile_set.shape)
        logger.debug("Checking neighbours of current tile: {}".format(unr_tile_ind))
        # Go through the 4 directly neighbouring tiles
        try:
            upper_neighbour = (unr_tile_ind[0] - 1, unr_tile_ind[1])
            upper_neighbour_ind = np.ravel_multi_index(upper_neighbour, tile_set.shape)
            if tiles[upper_neighbour_ind] is not None:
                # There is an upper neighbour
                border_direct[0] = true_int
                logger.debug("Direct neighbour top")
        except ValueError:
            # There is no upper neighbour
            pass
        try:
            left_neighbour_ind = (unr_tile_ind[0], unr_tile_ind[1] - 1)
            left_neighbour_ind = np.ravel_multi_index(left_neighbour_ind, tile_set.shape)
            logger.debug('neighbouring tile on left: {}'.format(tiles[left_neighbour_ind]))
            if tiles[left_neighbour_ind] is not None:
                # There is a left neighbour
                border_direct[1] = true_int
        except ValueError:
            pass
        try:
            lower_neighbour_ind = (unr_tile_ind[0] + 1, unr_tile_ind[1])
            lower_neighbour_ind = np.ravel_multi_index(lower_neighbour_ind, tile_set.shape)
            if tiles[lower_neighbour_ind] is not None:
                # There is a lower neighbour
                border_direct[2] = true_int
        except ValueError:
            pass
        try:
            right_neighbour_ind = (unr_tile_ind[0], unr_tile_ind[1] + 1)
            right_neighbour_ind = np.ravel_multi_index(right_neighbour_ind, tile_set.shape)
            if tiles[right_neighbour_ind] is not None:
                # There is a right neighbour
                border_direct[3] = true_int
        except ValueError:
            pass

        # Extra cases for indirect neighbours:
        try:
            indirect_neighbour_ind = (unr_tile_ind[0] - 1, unr_tile_ind[1] - 1)
            indirect_neighbour_ind = np.ravel_multi_index(indirect_neighbour_ind, tile_set.shape)
        except ValueError:
            pass
        else:

            if tiles[indirect_neighbour_ind] is not None:
                # There is an indirect neigbour on the top left
                border_indirect[0] = true_int
                border_indirect[1] = true_int
        try:
            indirect_neighbour_ind = (unr_tile_ind[0] - 1, unr_tile_ind[1] + 1)
            indirect_neighbour_ind = np.ravel_multi_index(indirect_neighbour_ind, tile_set.shape)
        except ValueError:
            pass
        else:
            if tiles[indirect_neighbour_ind] is not None:
                # There is an indirect upper neigbour on the right
                border_indirect[0] = true_int
                border_indirect[3] = true_int
        try:
            indirect_neighbour_ind = (unr_tile_ind[0] + 1, unr_tile_ind[1] - 1)
            indirect_neighbour_ind = np.ravel_multi_index(indirect_neighbour_ind, tile_set.shape)
        except ValueError:
            pass
        else:
            if tiles[indirect_neighbour_ind] is not None:
                # There is an indirect neigbour on the bottom left
                border_indirect[2] = true_int
                border_indirect[1] = true_int
        try:
            indirect_neighbour_ind = (unr_tile_ind[0] + 1, unr_tile_ind[1] + 1)
            indirect_neighbour_ind = np.ravel_multi_index(indirect_neighbour_ind, tile_set.shape)
        except ValueError:
            pass
        else:
            if tiles[indirect_neighbour_ind] is not None:
                # There is an indirect neigbour on the bottom right
                border_indirect[2] = true_int
                border_indirect[3] = true_int
        # Take the indirect and direct borders together:
        border_all = np.minimum(border_direct, border_indirect)
        logger.debug("Border direct: {}".format(border_direct))
        logger.debug("Border indirect: {}".format(border_indirect))
        logger.debug("Border all: {}".format(border_all))

        # For each pixel in the overlapping pixels calculate the distance from
        # the border and determine a weight based on the distance

        # Determine how many tiles overlap in the overlapping pixels
        overlapping_values = cur_mask[overlapping_inds]
        # Determine border distance to the top-left border and the bottom-right border:
        # So that there will be an np array of shape: (nr of pixels, 4), where each row
        # contains the distance to the corners in this order: [top, left, bottom, right]
        overlapping_pxls = np.array(overlapping_inds).T
        border_dist_list_raw = np.hstack((abs(overlapping_pxls - cur_corner_tl),
                                      abs(overlapping_pxls - cur_corner_br)))
        logger.debug("border_dist_list: {}".format(border_dist_list_raw))
        logger.debug("border_dist_list shape: {}".format(border_dist_list_raw.shape))

        # Take out irrelevant borders, add a large number so thay will not be taken
        # into account in the later analysis
        border_dist_list_indirect = border_dist_list_raw + border_indirect
        border_dist_list_direct = border_dist_list_raw + border_direct
        border_dist_list = border_dist_list_raw + border_all

        # Center of the tile
        center = int(nr_pixels / 2)
        logger.debug("center: {}".format(center))
        # Find all the pixels that have the center dist to one of the borders:
        # Center pixels will include pixels that form a plus centered
        # on the center of the tile and are part of an overlapping bit, like so:
        #  _________
        # |    |    |
        # |__     __|
        # |         |
        # |____|____|
        #

        center_pixel_inds = np.where(border_dist_list_raw == center)[0]
        logger.debug("center_pixel_inds: {}".format(center_pixel_inds))
        if center_pixel_inds.any():
            # Find the normalization factor for each border:
            # Look at the border distance of the pixels in the center
            center_pixels_dist = border_dist_list_direct[center_pixel_inds, :]
            # Get only the center pixels distances that belong to each border,
            # prevents taking distances into account that are to far to belong to
            # the corresponding border.
            # Set the pixels that are farther away than the center to 1,
            # to prevent them from being included in further analysis.
            center_pixels_dist[center_pixels_dist >= center] = 1
            # Find the maximum distance to each border within the center pixels.
            # This will be used as the maximum distance to the border, within
            # the overlap that belongs to that border.
            # (yields array of length 4)
            max_dist_pixels = np.amax(center_pixels_dist, axis = 0)
            logger.debug("Max border dist pixels: {}".format(max_dist_pixels))
            # Normalize with the found maximum distance for each border:
            border_dist_list_direct_norm = np.asfarray(border_dist_list_direct) \
                                                    / np.asfarray(max_dist_pixels)
            logger.debug("Closest border for each pixel: {}"
                            .format(np.argmin(border_dist_list_direct_norm,  axis = 1)))
            # Get the normalized distance to the closest border for each pixel.
            # We now have a 1D array of length nr of pixels, where each entry
            # represents the normalized distance from the pixel to the border
            # it belongs to.
            border_dist_norm = np.nanmin(border_dist_list_direct_norm, axis = 1)
            logger.debug("border_dist_norm: {}".format(border_dist_norm))
            logger.debug("border_dist_norm max, min: {} {}"
                            .format(np.amax(border_dist_norm),
                                    np.amin(border_dist_norm)))

            # Plots to check which pixels belong to which border,
            # uses closest_border.
            #closest_border = np.argmin(border_dist_list_direct_norm, axis=1)
            #cur_temp_mask1 = np.copy(cur_mask)
            #cur_temp_mask1[:,:] = 4.0
            #cur_temp_mask1[overlapping_inds] = np.array(closest_border)
            #plt.figure("Assigned borders")
            #plt.imshow(cur_temp_mask1, interpolation = 'none')
            #plt.show(block = False)
            #cur_temp_mask2 = np.copy(cur_mask)
            #cur_temp_mask2[overlapping_inds] = np.array(border_dist_norm)
            #plt.figure("border distance")
            #plt.imshow(cur_temp_mask2, 'gray', interpolation = 'none')
            #plt.show()

            # Plot the mask and blended tile
            # cur_temp_mask = np.copy(cur_mask)
            # cur_temp_mask[overlapping_inds] = np.array(border_dist_norm)
            # plt.figure("Normalized distance linear")
            # plt.imshow(cur_temp_mask, 'gray', interpolation = 'none')
            # plt.show()

            # Now calculate weights for the blending with direct neighbours
            if linear_blending:
                weights = border_dist_norm
            else:
                weights = non_linear_blending(border_dist_norm)
            logger.debug("weights max, min: {} {}"
                            .format(np.amax(weights), np.amin(weights)))
            # Plot the mask and blended tile
            # cur_temp_mask = np.copy(cur_mask)
            # cur_temp_mask[overlapping_inds] = np.array(weights)
            # plt.figure("Blending without corner blending")
            # plt.imshow(cur_temp_mask, 'gray', interpolation = 'none')
            # plt.show()
            #~ plt.figure("Blending tile")
            #~ plt.imshow(cur_tile, 'gray', interpolation = 'none')
            #~ plt.show()

            # Change the weights in the corners:
            weights = check_corner_blending(weights, border_dist_list, max_dist_pixels,
                            overlapping_values,
                            border_direct, border_indirect, center,
                            linear_blending)
        else:
            # We have only a corner overlapping, no sides
            logger.debug('{} has only a lonely corner and nothing else'.format(tile_ind))
            # Set weights to one to give them to check_corner_blending
            weights = np.ones(len(overlapping_inds[0]))
            # Blend the corner, with max_value set to two.
            weights = check_corner_blending(weights, border_dist_list, None,
                            overlapping_values,
                            border_direct, border_indirect, center,
                            linear_blending, max_value = 2)
        
        # Converted to float because data where save in uint16
        cur_tile=img_as_float(cur_tile)
        # Apply the weights to the pictures
        if cur_tile.ndim == 3:
            for i in range(cur_tile.shape[0]):
                cur_tile[i][overlapping_inds] *= np.array(weights)
                #~ plt.figure("Blending tile")
                #~ plt.imshow(cur_tile[i], 'gray', interpolation = 'none')
                #~ plt.show()
        else:
            cur_tile[overlapping_inds] *= np.array(weights)

        # Fill the temporary mask for debugging
        cur_temp_mask = np.copy(cur_mask)
        cur_temp_mask[overlapping_inds] = np.array(weights)

        # Plot the mask and blended tile
        #~ if tile_ind in [32, 33, 34, 60, 61, 62, 88, 89, 90]:
            #~ plt.figure("Blending tile")
            #~ plt.imshow(cur_tile, 'gray', interpolation = 'none')
            #~ plt.show()
    else:
        # If there is no overlap just return the original tile and
        # the temporary mask for debugging
        cur_temp_mask = np.copy(cur_mask)

    return cur_tile, cur_temp_mask


def check_corner_blending(weights, border_dist_list, max_dist_pixels,
                          overlapping_values, border_direct,
                          border_indirect, center, linear_blending,
                          max_value = 4):
    """Function to improve blending of differently overlapping corners.

    In a corner 2, 3 or 4 tile can overlap. Depending on how many tiles
    overlap and in wich direction they overlap the weights for a tile
    are adjusted by this function.

    Parameters:
    -----------

    weights: np.array
        1D numpy array. A weight for each overlapping
        pixel, as determined by the function
        perform_blending_par_proof (Weights depend on
        direct borders, or 1 if there is only a lonely
        corner)
    border_dist_list: np.array
        2D numpy array of shape: (nr of overlapping
        pixels, 4). Array containing for each pixel the
        distance to each of the four borders in this
        order [top, left, bottom, right], for borders
        that do not overlap to another
        tile the distance has been set to 999 or
        greater.
    max_dist_pixels: np.array
        Numpy array of shape: (1, 4). The maximum
        distance to each of the four borders in this
        order [top, left, bottom, right], for borders
        that do not overlap to another tile the max
        distance will be to 9999 or greater.
    overlapping_values: np.array
        1D numpy array of floats. An float value
        indicating how many tiles are overlapping
        for each overlapping pixel. This array has the
        same size as weights.
    border_direct: np.array
        1 by 4 numpy array. Array indicating for each
        border if there is a direct overlap.
        If there is overlap the value is 0, otherwise
        the value is 9999.
    border_indirect: np.array
        1 by 4 numpy array. Array indicating for each
        border if there is a indirect overlap.
        If there is overlap the value is 0, otherwise
        the value is 9999.
    center: int
        The center of th tile, this is the
        same number in the x and y direction.
    linear_blending: bool
        If true perform linear blending,
        otherwise non-linear blending.
    max_value: int
        The maximum value expected in the
        overlap. Default is 4, this is the overall
        maximum in the blending mask. When max_value
        is 2, we assume that the tile only overlaps
        in one or multiple corners. Because when only a
        corner of a tile is overlapping and there is
        no other overlap the maximum is 2. (Default: 4)

    Returns:
    --------

    weights: np.array
        1D numpy array. The array 'weights' that was passed
        to as an argument, but now adjusted for the corners.
    """
    logger.info("Performing corner blending...")
    # Find the corners
    # Define the borders that belong to each corner
    corner_defs = [[0,1], [0,3], [2,1],[2,3]]
    logger.debug("Border_dist_list: {}".format(border_dist_list[:,:]))
    for corner in corner_defs:
        # Init lonely corner
        lonely_corner = False
        # Find the indexes of the corner pixels
        corner_inds1 = np.where(border_dist_list[:,corner[0]] <= center)
        corner_inds2 = np.where(border_dist_list[:,corner[1]] <= center)
        big_corner_inds  = np.intersect1d(corner_inds1[0], corner_inds2[0])
        # Print the corners indexes
        logger.debug("Corner {} inds1: {}".format(corner, corner_inds1))
        logger.debug("Corner {} inds2: {}".format(corner, corner_inds2))
        logger.debug("Corner {} inds_intersect: {}".format(corner, big_corner_inds))
        # Determine if there is a corner where only 2 tiles overlap
        # (if this is the case lonely_corner will be set to True)
        # If 3 or four tiles overlap lonely corner is set to False.
        if ((border_indirect[corner[0]] < 999) and (border_direct[corner[0]] > 1)
            and (border_indirect[corner[1]] < 999) and (border_direct[corner[1]] > 1)):
            # There are only indirect tiles overlapping with this corner
            pos_corner_inds = np.where(overlapping_values == 2.0)[0]
            lonely_corner = True
        elif max_value == 2:
            # If max_value of 2 was passed, assume that the corner has
            # only 2 tiles overlapping
            pos_corner_inds = np.where(overlapping_values == 2.0)[0]
            lonely_corner = True
        else:
            # Else assume that more tiles are overlapping
            pos_corner_inds = np.where(overlapping_values > 2.0)[0]
        corner_inds  = np.intersect1d(big_corner_inds, pos_corner_inds)
        logger.debug("Corner {} inds_intersect pos: {}".format(corner, corner_inds))

        #If we found any overlap in the corner:
        if corner_inds.any():
            if lonely_corner:
                # 2 tiles are overlapping, but only in the corner
                # Calculate the Manhattan distance to the corner of the
                # current tile for each pixel belonging to this corning.
                border_dist_corner =  np.asfarray(border_dist_list[corner_inds, corner[0]] \
                                    + border_dist_list[corner_inds, corner[1]])
                logger.debug('LC border_dist_corner {}'.format(border_dist_corner))
                # Find the maximum distance for normalization
                max_dist_pixels_corner = float(max(border_dist_corner))
                logger.debug('LC max_dist_pixels_corner {}'.format(max_dist_pixels_corner))
                # Normalize and apply non-linear function if necessary
                if linear_blending:
                    weights_corner = border_dist_corner / \
                                        max_dist_pixels_corner
                else:
                    weights_corner = non_linear_blending(
                                            border_dist_corner / \
                                            max_dist_pixels_corner)
                #Place back new weights, overwrite old:
                weights[corner_inds] = weights_corner
                logger.debug('LC weights {}'.format(weights))
                logger.debug('LC max weights {}'.format(max(weights)))
            else:
                if np.mean(overlapping_values[corner_inds]) >= 3.5:
                    # 4 tiles overlap.
                    # If the mean of this corner is larger than 3.5, assume
                    # that 4 tiles overlap here.

                    # Pick the indexes of the corner where all 4 tiles
                    # actually overlap
                    corner_inds4 = np.where(overlapping_values == 4)
                    corner_inds  = np.intersect1d(corner_inds, corner_inds4[0])
                    logger.debug("Corner {} inds_intersect  4: {}".format(corner, corner_inds))
                    #Find the manhattan distance to corner at these indexes
                    border_dist_corner = border_dist_list[corner_inds, corner[0]] \
                                         + border_dist_list[corner_inds, corner[1]]
                    max_dist_pixels_corner = float(max(border_dist_corner))
                    # Normalize and apply non-linear function if necessary
                    if linear_blending:
                        weights_corner = border_dist_corner / \
                                            max_dist_pixels_corner
                    else:
                        y = np.asfarray(border_dist_list[corner_inds, corner[0]]) / \
                                                float(max_dist_pixels[corner[0]])
                        x = np.asfarray(border_dist_list[corner_inds, corner[1]]) / \
                                                float(max_dist_pixels[corner[1]])
                        logger.debug('x: {}  x max: {}'.format(x, max(x)))
                        logger.debug('y: {} y max: {}'.format(y, max(y)))
                        weights_corner = non_linear_blending_corner(x, y)
                    # Place back new weights:
                    weights[corner_inds] = weights_corner
                elif np.mean(overlapping_values[corner_inds]) < 3.5:
                    # 3 corners overlap.
                    # If the mean of this corner is smaller than 3.5, assume
                    # that 4 tiles overlap here.
                    corner_inds3    = np.intersect1d(corner_inds, np.where(overlapping_values == 3)[0])
                    #corner_inds     = corner_inds3
                    logger.debug("Corner {} inds_intersect 3: {}".format(corner, corner_inds3))
                    # Find which of the two borders is not a direct border:
                    special_border = None
                    special_border_list = np.where(border_direct > 1)[0]
                    logger.debug('special_border_list {}'.format(special_border_list))
                    for border in corner:
                        if border in special_border_list:
                            logger.debug('special_border found {}'.format(border))
                            special_border = border
                    if special_border is not None:
                        # Special border means that this border does
                        # not overlap with the tile next to it,
                        # therefore there are no overlapping pixels
                        # assigned to this border.
                        # Take all pixels that belong to the special
                        # border into account with the corner:
                        # Determine the maximum distance from the
                        # borders within this corner
                        max_dist_pixels_corner = np.amax(border_dist_list[corner_inds3, :], axis = 0)
                        # Determine new corner pixels based on the maximum distance
                        extra_corner_inds1 = np.where(border_dist_list[:, special_border]
                                                    <= max_dist_pixels_corner[special_border])[0]
                        logger.debug('extra_corner_inds1 {}'.format(extra_corner_inds1))
                        logger.debug('big_corner_inds {}'.format(big_corner_inds))
                        #Replace corner inds with new corner indexes, including the extra corner indexes:
                        corner_inds = np.intersect1d(extra_corner_inds1, big_corner_inds)
                        logger.debug('corner_inds shape {}'.format(corner_inds.shape))
                        # Replace corner inds 3, with part of the corner that is 3:
                        corner_inds3 = np.where(overlapping_values[corner_inds] == 3)[0]
                        logger.debug('corner_inds3 shape {}'.format(corner_inds3.shape))

                        # Get the manhattan distance where 3 tiles overlap
                        border_dist_corner3 = np.asfarray(
                                border_dist_list[corner_inds[corner_inds3], corner[0]] \
                                + border_dist_list[corner_inds[corner_inds3], corner[1]])
                        logger.debug('border_dist_corner {}'.format(border_dist_corner3))
                        max_man_dist_corner = float(max(border_dist_corner3))
                        logger.debug('max_man_dist_corner {}'.format(max_man_dist_corner))
                        # Also get the manhattan distance for all corner pixels,
                        # also where only 2 tiles overlap
                        border_dist_corner = np.asfarray(
                                    border_dist_list[corner_inds, corner[0]] \
                                    + border_dist_list[corner_inds, corner[1]])
                        # Then normalize this using the maximum of the manhattan
                        # distance in the area where 3 tiles overlap.
                        # So the part where 3 tiles overlap will be normalized in
                        # a normal way, but the parts where only 2 tiles overlap will
                        # not be normalized properly.
                        norm_manh_dist = border_dist_corner / max_man_dist_corner
                        # Then, adjust the norm_manh_dist to deal with the parts where two tiles overlap.
                        # Get the minimum and maximum border distance where all 3 tiles overlap
                        # (direct distance to the closest border)
                        max_corner3_dist = np.amax(
                                            border_dist_list[corner_inds[corner_inds3],:], axis = 0)
                        min_corner3_dist = np.amin(
                                            border_dist_list[corner_inds[corner_inds3],:], axis = 0)
                        logger.debug('max_corner3_dist min_corner3_dist {} {}'
                                    .format(max_corner3_dist, min_corner3_dist))
                        # Then used this min and max to adjust the weights of pixels that are closer to a border
                        # to zero and weights of pixels that are farther away from a border to 1.
                        for i in corner:
                            norm_manh_dist[border_dist_list[corner_inds, i]
                                                    < min_corner3_dist[i]] = 0.0
                            logger.debug('number of pixels closer to corner than end overlap 3: {}'
                                         .format(len(norm_manh_dist[border_dist_list[corner_inds, i]
                                                    < min_corner3_dist[i]])))
                            norm_manh_dist[border_dist_list[corner_inds, i]
                                                    > max_corner3_dist[i]] = 1.0
                            logger.debug('number of pixels farther away from corner than start overlap 3: {}'
                                     .format(len(norm_manh_dist[border_dist_list[corner_inds, i]
                                                    > max_corner3_dist[i]])))
                        # Make the weights non-linear if necessary
                        if linear_blending:
                            corner_weights = norm_manh_dist
                        else:
                            corner_weights = non_linear_blending(norm_manh_dist)
                        # Multiply the original weights in the corner with the  newly calculated corner weights,
                        # this way in the corner we get a combination of weights decreasing towards the direct border
                        # and weights decreasing towards the corner of the tile.
                        weights[corner_inds] *= corner_weights

    return weights


def make_final_image(joining, temp_file, im_file, nr_pixels):
    """Puts blended tiles at the correct position in the final image.

    Takes blended tiles as found in temp_file and "pastes" them at the
    correct position as indicated by joining in the final image. The
    final image is kept in im_file.

    Parameters:
    -----------

    joining: dict
        Containing corners for tiles
    temp_file: pointer
        Pointer to hdf5 object with the following groups:
        tiles,  temp_masks, ubytes.
        Each group contains as many datasets as there are
        tiles, the datasets are named after the the tile
        index found in the first element of corner.
        This function places a blended tile and a corner in
        data set that matches the tile ind argument.
    im_file: pointer
        Pointer to hdf5 object with dataset
        "blending_mask"
        which contains a numpy array. blending_mask should
        be 1 where ther is no overlap and 2, 3 or 4 where
        the respective number of tiles overlap.
        Other datasets in this file are: final_image
        and temp_mask
    nr_pixels: int
        Size of the tile, tile is assumed to be a square.
    """
    logger.info("Looping over all tiles and pasting them in the final "
                + "image...")
    for i, corner in joining['corner_list']:
        if not (np.isnan(corner[0])):  # Check for empty tile
            logger.info('Placing tile: {} in corner: {}'
                        .format(i, corner))
            # Pick the right region of the image using the corner
            ymin = int(corner[0])
            ymax = int(corner[0]) + nr_pixels
            xmin = int(corner[1])
            xmax = int(corner[1]) + nr_pixels
            logger.debug('i: {} ymin: {} ymax: {} xmin: {} xmax: {}'.
                         format(i, ymin, ymax, xmin, xmax))
            # Place the tile:
            if len(joining['final_image_shape']) == 3:
                im_file['final_image'][:, ymin:ymax, xmin:xmax] += temp_file['blended_tiles'][str(i)]
                if 'final_image_ubyte' in im_file:
                    im_file['final_image_ubyte'][:, ymin:ymax, xmin:xmax] += temp_file['ubytes'][str(i)]
            else:
                im_file['final_image'][ymin:ymax, xmin:xmax] += temp_file['blended_tiles'][str(i)]
                if 'final_image_ubyte' in im_file:
                    im_file['final_image_ubyte'][ymin:ymax, xmin:xmax] += temp_file['ubytes'][str(i)]
            im_file['temp_mask'][ymin:ymax, xmin:xmax] += temp_file['temp_masks'][str(i)]

def paste_in_final_image_MPI(joining, temp_file, im_file, tile_ind, nr_pixels):
    """Puts one blended tile at the correct position in the final image.

    Takes the blended tile as found at tile_ind in temp_file and
    "pastes" them at the correct position as indicated by joining in
    the final image. The final image is kept in im_file.

    Parameters:
    -----------

    joining: dict
        Containing corners for tiles
    temp_file: pointer
        Pointer to hdf5 object with the following groups:
        tiles,  temp_masks, ubytes.
        Each group contains as many datasets as there are
        tiles, the datasets are named after the the tile
        index found in the first element of corner.
        This function places a blended tile and a corner in
        data set that matches the tile ind argument.
    im_file: pointer
        Pointer to hdf5 object with dataset
        "blending_mask"
        which contains a numpy array. blending_mask should
        be 1 where ther is no overlap and 2, 3 or 4 where
        the respective number of tiles overlap.
        Other datasets in this file are: final_image
        and temp_mask
    tile_ind: int
        Index of the tile that should be placed
    nr_pixels: int
        Size of the tile, tile is assumed to be a square.
    """
    # Find the right corner
    cur_corner = next(([i, corner] for i, corner in joining['corner_list']
                        if (i == tile_ind)), [0, [np.nan, np.nan]])[1]
    logger.info('Pasting tile {} in final image, in corner: {}'
                .format(tile_ind, cur_corner))
    logger.debug('tile_ind: {} cur_corner: {}'.format(tile_ind, cur_corner))
    if not (np.isnan(cur_corner[0])):            # Check for empty tile
        # Pick the right region of the image using the corner
        ymin = int(cur_corner[0])
        ymax = int(cur_corner[0]) + nr_pixels
        xmin = int(cur_corner[1])
        xmax = int(cur_corner[1]) + nr_pixels
        logger.debug('Global corners: {} {} {} {}'.
                     format(ymin, ymax, xmin, xmax))
        # Place the tile:
        if len(joining['final_image_shape']) == 3:
            im_file['final_image'][:, ymin:ymax, xmin:xmax] += temp_file['blended_tiles'][str(tile_ind)]
            if 'final_image_ubyte' in im_file:
                im_file['final_image_ubyte'][:, ymin:ymax, xmin:xmax] += temp_file['ubytes'][str(tile_ind)]
        else:
            im_file['final_image'][ymin:ymax, xmin:xmax] += temp_file['blended_tiles'][str(tile_ind)]
            if 'final_image_ubyte' in im_file:
                im_file['final_image_ubyte'][ymin:ymax, xmin:xmax] += temp_file['ubytes'][str(tile_ind)]
        if 'temp_mask' in im_file:
            im_file['temp_mask'][ymin:ymax, xmin:xmax] += temp_file['temp_masks'][str(tile_ind)]


def assess_overlap(joining, tiles, tile_file, contig_tuples):
    """Calculate the covariance of the overlap

    This function is usefull to see the quality of the final overlap.
    It always works with flattened (max projected) tiles.

    Parameters:
    -----------

    joining: dict      
        Dictionary containing the corner list
        (with key: 'corner_list') with the tile indexes
        and their corresponding corners
    tiles: list
        List of hdf5-references, references
        to the images in tile_file, with a
        reference for each tile.
        If a tile does not have an associated image,
        its name is None.
    tile_file: pointer
        hdf5 file object. The opened file containing the
        tiles to stitch.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.

    Returns:
    --------

    xcov: float
        The cross covariance between the overlapping parts of the two images.
    """
    logger.info("Assessing overlap...")
    allow_plot = False
    xcov_list = []
    sorted_corners = sorted(joining['corner_list'])
    logger.debug('sorted corner list: {}'.format(sorted_corners))
    for pair in contig_tuples:
        xcov = np.nan
        ind1 = min(pair)
        ind2 = max(pair)
        if tiles[ind1] is not None and tiles[ind2] is not None:
            if abs(ind1 - ind2) == 1:
                #Overlap on left or right
                logger.debug(("Overlap on right of tile {0} and left of"
                                    + " tile {1}").format(ind1, ind2))
                corner1 = sorted_corners[ind1][1]
                corner2 = sorted_corners[ind2][1]
                #Calculate overlap indexes
                logger.debug("x values of, ind {}, x-coord: {}  ind {}, x-coord: {}"
                                .format(ind1,  corner1[1], ind2, corner2[1]))
                logger.debug("y values of, ind {}, x-coord: {}  ind {}, x-coord: {}"
                                .format(ind1,  corner1[0], ind2, corner2[0]))
                overlap_ind_x   = int(corner1[1] - corner2[1])
                overlap_ind_y   = int(corner1[0] - corner2[0])
                logger.debug("Overlap index, x: {} y: {}".
                                format(overlap_ind_x, overlap_ind_y))
                # Loading tiles in function call, loading the flattened version of the tiles.
                overlap1, overlap2 = ph.get_overlapping_region(inout.load_tile(tiles[ind1], tile_file),
                                        inout.load_tile(tiles[ind2], tile_file),
                                        overlap_ind_x, overlap_ind_y, 'left')
                #For subplotting the overlaps later:
                plot_order = np.ones((1,3))
            else:
                #Overlap on top or bottom
                logger.debug(("Overlap on bottom of tile {0} and top of"
                                    + "tile {1}").format(ind1, ind2))
                corner1 = sorted_corners[ind1][1]
                corner2 = sorted_corners[ind2][1]
                #Calculate overlap indexes
                logger.debug("x values of, ind {}, x-coord: {} cur_corner ind {}, x-coord: {}"
                                .format(ind1,  corner1[1], ind2, corner2[1]))
                logger.debug("y values of, ind {}, y-coord: {}  ind {}, y-coord: {}"
                                .format(ind1,  corner1[0], ind2, corner2[0]))
                overlap_ind_x   = int(corner1[1] - corner2[1])
                overlap_ind_y   = int(corner1[0] - corner2[0])
                logger.debug("Overlap index, y: {} x: {}".
                                format(overlap_ind_y, overlap_ind_x))
                overlap1, overlap2 = ph.get_overlapping_region(inout.load_tile(tiles[ind1], tile_file),
                                        inout.load_tile(tiles[ind2], tile_file),
                                        overlap_ind_x, overlap_ind_y, 'top')
                #For subplotting the overlaps later:
                plot_order = np.ones((3,1))


            if allow_plot:
                overlap_rgb         = np.zeros((overlap1.shape[0], overlap1.shape[1], 3))
                overlap_rgb[:,:,0]  = smex.rescale_intensity(overlap1)
                overlap_rgb[:,:,1]  = smex.rescale_intensity(overlap2)

                #overlap_masked          = np.copy(overlap_rgb)
                #overlap_masked[:,:,0][overlapping_inds]    *=  1.0 - np.array(weights)
                #overlap_masked[:,:,1][overlapping_inds]    *= np.array(weights)

            cor_coeff, mono_color = ph.xcov_nd(overlap1, overlap2)
            if not(mono_color):
                xcov = cor_coeff

            if mono_color:
                logger.debug("The overlapping images are mono colored, "
                             + "xcov: {}".format(cor_coeff))
                title_text = "Mono color, xcov: {}".format(cor_coeff)
            else:
                logger.debug("Normalized cross covariance of overlap: "
                             + "{}".format(cor_coeff))
                title_text = "XCov: {}".format(cor_coeff)

            if allow_plot:
                inout.display_tiles([overlap1,
                                overlap2,
                                overlap_rgb],
                                plot_order, fig_nr = "overlap check",
                                maximize = True, main_title = title_text,
                                rgb = True)
        xcov_list.append(xcov)
    return xcov_list


def generate_blended_tile_npy(corner_ind_coord,stitching_files_dir,
                              blended_tiles_directory, masked_tiles_directory,
                              analysis_name,processing_hyb,reference_gene,
                              micData, tiles, nr_pixels,
                linear_blending):

    
    """
    Blend the tile if necessary and then save it temp blended folder.
    Modification of the generate_blended_tile that run using .npy files
    and doesn't save the data in a hdf5 file

    Parameters:
    ------------
    
    corner_ind_coord: list
        Contains two elements, the first one is an
        int representing the tile index, the second one is
        a numpy array containing the corner's coordinates.
    stitching_files_dir: str
        Path to the files to stitch
    analysis_name: str 
        Name of the current analysis
    blended_tiles_directory: str
        Path to the directory where to save the blended tiles    
    masked_tiles_directory: str
        Path to the directory with the masks                                       
    processing_hyb: str
        Name of the hybridization processed
    reference_gene: str
        Name of the gene to be stitched
    blending_mask: np.array
        Array containing the blending mask used for blending the images     
    micData: object
        MicroscopeData object. Contains coordinates of
        the tile corners as taken from the microscope.                   
    tiles: np.array
        Array with tile number. -1 correspond to missing tile      
    nr_pixels: int
        Denoting size of the tile. 
    linear_blending: bool
        When True blending will be linear and when False, blending will be non-linear.
                    
    
    
    """
       
    stitching_files_list = glob.glob(stitching_files_dir+'*.npy')
    masked_files_list = glob.glob(masked_tiles_directory+'*.npy')
    
    # Get corner
    tile_ind = corner_ind_coord[0]
    corner = corner_ind_coord[1]
    
    # Get the reference address of the tile to load
    tile_ref = tiles[tile_ind]
    
    # Check if the tile is existing
    if tile_ref != -1:
        
        # Get the tile number
        tile_number = '_'+str(tile_ref)+'.npy'
        
        # Load the tile
        tile_path = [tile_p for tile_p in stitching_files_list if tile_number in tile_p][0]
        cur_tile = np.load(tile_path)
        
        # Load the mask
        mask_ref = '_'+str(tile_ind)+'.npy'
        mask_path = [mask_p for mask_p in stitching_files_list if mask_ref in mask_p]
        if mask_path:
            cur_mask = np.load(mask_path[0])
        
            #Blend
            blended_tile, cur_temp_mask  = perform_blending_par_proof(tile_ind,
                    cur_mask, cur_tile,
                    linear_blending, tiles, micData.tile_set, nr_pixels)
        
        else:
            blended_tile = cur_tile
        # Convert the blended tile to uint16 to be able to save it in the 
        # prefilled hdf5
        blended_tile = img_as_uint(blended_tile)

        # Save the blended tiles
        fname = blended_tiles_directory + analysis_name+'_'+processing_hyb+'_'+reference_gene+'_blended_tile_pos'+tile_number
        np.save(fname,blended_tile)


# Write the blended image in the hdf5 file
def make_final_image_npy(joining, stitching_file, blended_tiles_directory, tiles,gene, nr_pixels):
    
    """
    Puts blended tiles at the correct position in the final image.
    Modified version of the make_final_image that works on .npy stored
    images. Works only in 2D.

    Takes blended tiles as found in the blended_tiles_directory and "pastes" 
    them at the correct position as indicated by joining in the final image
    in the hdf5 file. 

    Parameters:
    -----------

    joining: dict
        Containing corners for tiles
    stitching_file: pointer
        Pointer to hdf5 object with the following groups:
    blended_tiles_directory: str
        Path to the directory where to save the blended tiles                    
    tiles: np.array
        Array with tile number. -1 correspond to missing tile
    gene: str
        Name of the gene to be stitched
    nr_pixels: int
        Size of the tile, tile is assumed to be a square.
    
    """
    
    
    blended_files_list = glob.glob(blended_tiles_directory+'*.npy')
    
    for i, corner in joining['corner_list']:
         
            tile_ref = tiles[i]
            
            if tile_ref != -1:
            # Get the tile number
                
                tile_number = '_'+str(tile_ref)+'.npy'
            
                image_path = [ipath for ipath in blended_files_list if tile_number in ipath][0]

                img = np.load(image_path)

                # Pick the right region of the image using the corner
                ymin = int(corner[0])
                ymax = int(corner[0]) + nr_pixels
                xmin = int(corner[1])
                xmax = int(corner[1]) + nr_pixels

                # Place the tile:
                stitching_file[gene]['StitchedImage']['final_image'][ymin:ymax, xmin:xmax] += img          
                stitching_file.flush()