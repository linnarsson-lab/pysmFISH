"""High level helper functions that deal with the alignment of
a single pair of neighbours, written for use by the class pairwise
alignment.
Creates its own logger object when imported.

Functions:
align_single_pair       -- Determine the ideal alignment between two
                        neighbouring tiles
refine_single_pair      -- Determine the ideal alignment between two
                        neighbouring tiles, with the use of an old
                        alignment.
determine_overlap       -- Determine the overlap between two
                        neighbouring tiles
calculate_pos_shifts    -- Calulate possible shifts, given two
                        overlapping images.
find_best_trans         -- Find the best translation using the cross
                        covariance.
find_best_trans_corr    -- Find the best translation using the cross
                        correlation.
perform_upsampling      -- Perform upsampling for subpixel precision in
                        the shift
"""

import numpy as np
from skimage import img_as_float
# import matplotlib.pyplot as plt
import logging

from . import inout
from . import pairwisehelper as ph


logger = logging.getLogger(__name__)

# The method used for the 3D alignment.
# Options are: 'compress pic', 'use whole pic' and 'calculate per layer'.
# We have been using 'compress pic', which  is the most accurate and
# fastest method.
# Therefore  'use whole pic' and 'calculate per layer' have not been
# extensively tested.
method = 'compress pic'


#######################Second level functions#######################
def align_single_pair(tiles, tile_file, contig_tuples, contig_ind,
                      micData, nr_peaks, nr_dim=2,
                      nr_slices=None):
    """Determine the ideal alignment between two neighbouring tiles

    Parameters:
    -----------
    tiles: list
        List of strings. List of references to the the tiles in the hdf5 file tile_file.
    tile_file: pointer
        hdf5 file object. The opened file containing the tiles to stitch.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    contig_ind: int
        The index of the current tile pair in
        contig_tuples. More precisely: the index of the
        tuple in contig_tuples containing the indexes of
        the tiles that should be aligned.
    micData: object
        MicroscopeData object. Should contain coordinates of
        the tile corners as taken from the microscope.
        These coordinates are used to dtermine the overlap
        between a tile pair.
    nr_peaks: int
        The n highest peaks from the PCM
        matrix that will be used to do crosscovariance
        with. A good number for 2D analysis is 8 peaks and
        good numbers for 3D with method ='compress pic' are
        6 or 9 peaks.
    nr_dim: int
        If 3, the code will assume three
        dimensional data for the tile, where z is the first
        dimension and y and x the second and third. For any
        other value 2-dimensional data is assumed.
        (default: 2)
    nr_slices: int
        Only applicable when running with 3D
        pictures and using 'compres pic' method. Determines
        the number of slices that are compressed together
        (compression in the z-direction). If None,
        all the slices are compressed together. Default:
        None

    Returns:
    --------

    best_trans: np.array
        1 by 2 or 3 array containing best found (z), y and x translation.
    best_cov: float
        The covariance of the overlap after translation of the overlap by best_trans.
    contig_ind: int
        The index of the used tile pair in
        contig_tuples. This is necessary to return when
        running on multiple processors/cores. More
        precisely: the index of the tuple in contig_tuples
        containing the indexes of the tiles that should be
        aligned.
    """

    # Rename tile indexes
    ind1 = min(contig_tuples[contig_ind])
    ind2 = max(contig_tuples[contig_ind])
    logger.info("Calculating pairwise alignment for indexes: {} and {}"
                .format(ind1, ind2))
    if nr_dim == 3:
        tile_1 = inout.load_tile_3D(tiles[ind1], tile_file)
        tile_2 = inout.load_tile_3D(tiles[ind2], tile_file)
    else:
        tile_1 = inout.load_tile(tiles[ind1], tile_file)
        tile_2 = inout.load_tile(tiles[ind2], tile_file)

    if (tile_1.size and tile_2.size):
        # Determine overlap
        overlap1, overlap2, plot_order = determine_overlap(ind1,
                                                           ind2, tile_1,
                                                           tile_2,
                                                           micData)
        logger.debug("Shape of overlap 1 and 2: {} {}"
                     .format(overlap1.shape, overlap2.shape))

        if nr_dim == 3 and method == 'compress pic':
            if nr_slices is None:
                nr_slices = overlap1.shape[0]
            best_trans, best_cov = align_single_compress_pic(overlap1,
                                                             overlap2,
                                                             nr_peaks,
                                                             nr_dim,
                                                             nr_slices,
                                                             plot_order)
        else:
            unr_pos_transistions = calculate_pos_shifts(
                overlap1, overlap2, nr_peaks, nr_dim)
            logger.debug("Possible transistions: {}".
                         format(unr_pos_transistions))
            # Do correlation over the found shifts:
            best_trans, best_cov = find_best_trans(unr_pos_transistions,
                                                   overlap1, overlap2,
                                                   plot_order)

        # Give some feedback
        logger.info("Best shift: {} covariance: {}".format(best_trans,
                                                           best_cov))
        logger.debug(
            "Best shift type: {} {} covariance type: {} contig_ind type: {}"
            .format(type(best_trans), type(best_trans[0]),
                    type(best_cov), type(contig_ind)))
    else:
        # One of the tiles is empty
        best_trans = np.zeros(nr_dim, dtype=int)
        best_cov = np.nan
        logger.info(
            "Best shift: {}. One of the neighbours is empty".format(
                best_trans))
    return np.array(best_trans,dtype='int16'), best_cov, contig_ind


def refine_single_pair(tiles, tile_file, contig_tuples, contig_ind,
                       micData, old_P, nr_peaks,
                       nr_dim=2, nr_slices=None):
    """Determine the ideal alignment between two neighbouring tiles.

    Uses an old alignment as starting point. Meant to use on smFISH
    signal data, where the old alignment is taken from the aligning of
    the nuclei staining.

    Parameters:
    -----------

    tiles: list
        List of strings. List of references to the the tiles in the hdf5 file tile_file.
    tile_file: pointer
        hdf5 file object. The opened file containing the tiles to stitch.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    contig_ind: int
        The index of the current tile pair in
        contig_tuples. More precisely: the index of the
        tuple in contig_tuples containing the indexes of
        the tiles that should be aligned.
    micData: object
        MicroscopeData object. Should contain coordinates of
        the tile corners as taken from the microscope.
        These coordinates are used to dtermine the overlap
        between a tile pair.
    old_P: dict
        An old pairwise alignment containing a key
        'P' containing a flattened list of 2D or 3D
        pairwise translations.
        And containing a key 'covs' containing the
        normalized cross covariance for each alignment.
    nr_peaks: int
        The n highest peaks from the PCM
        matrix that will be used to do crosscovariance
        with. A good number for 2D analysis is 8 peaks and
        good numbers for 3D with method ='compress pic' are
        6 or 9 peaks.
    nr_dim: int
        If 3, the code will assume three
        dimensional data for the tile, where z is the first
        dimension and y and x the second and third. For any
        other value 2-dimensional data is assumed.
        (default: 2)
    nr_slices: int
        Only applicable when running with 3D
        pictures and using 'compres pic' method. Determines
        the number of slices that are compressed together
        (compression in the z-direction). If None,
        all the slices are compressed together. Default:
        None

    Returns:
    --------

    best_trans: np.array
        1 by 2 or 3 array containing best found (z), y and x translation.
    best_cov: float
        The covariance of the overlap after translation of the overlap by best_trans.
    contig_ind: int
        The index of the used tile pair in
        contig_tuples. This is necessary to return when
        running on multiple processors/cores. More
        precisely: the index of the tuple in contig_tuples
        containing the indexes of the tiles that should be
        aligned.
    """

    # Rename the indexes of the 2 images to compare:
    ind1 = min(contig_tuples[contig_ind])
    ind2 = max(contig_tuples[contig_ind])
    logger.info("Calculating pairwise alignment for indexes: {} and {}"
                .format(ind1, ind2))
    if nr_dim == 3:
        tile_1 = inout.load_tile_3D(tiles[ind1], tile_file)
        tile_2 = inout.load_tile_3D(tiles[ind2], tile_file)
    else:
        tile_1 = inout.load_tile(tiles[ind1], tile_file)
        tile_2 = inout.load_tile(tiles[ind2], tile_file)


    if (tile_1.size and tile_2.size):
        # Determine overlap
        overlap1, overlap2, plot_order = determine_overlap(ind1, ind2,
                                                           tile_1,
                                                           tile_2,
                                                           micData)
        # Pick the old translation from te pairwise alignment
        # array
        trans = old_P[contig_ind * nr_dim: contig_ind * nr_dim + nr_dim]
        logger.debug("Cur old trans is: {}".format(trans))

        # Determine the overlap translated according to the old
        # pairwise translation
        overlap1_ref, overlap2_ref = ph.calc_translated_pics(
            trans,
            overlap1, overlap2,
            round_size=True)
        # Take just a bright part from this overlap.
        # TODO: code below may crash when possible transistions
        # are too big.
        # cut_out_coord  = ph.select_cut_out2(overlap1_ref, overlap2_ref)
        # overlap1_ref = overlap1_ref[cut_out_coord[0]:(cut_out_coord[0] + 150),
        #                    cut_out_coord[1]:(cut_out_coord[1] + 150)]
        # overlap2_ref = overlap2_ref[cut_out_coord[0]:(cut_out_coord[0] + 150),
        #                    cut_out_coord[1]:(cut_out_coord[1] + 150)]
        logger.debug("Shape of overlap 1 and 2: {} {}"
                     .format(overlap1_ref.shape, overlap2_ref.shape))

        if nr_dim == 3 and method == 'compress pic':
            if nr_slices is None:
                nr_slices = overlap1.shape()[0]
            best_trans, best_cov = align_single_compress_pic(
                overlap1_ref,
                overlap2_ref, nr_peaks, nr_dim, nr_slices,
                plot_order)
        else:
            # Calculate possible transistions
            unr_pos_transistions = calculate_pos_shifts(overlap1_ref,
                                                        overlap2_ref,
                                                        nr_peaks,
                                                        nr_dim)
            # Calculate best xcov of the found shifts:
            best_trans, best_cov = find_best_trans(unr_pos_transistions,
                                                   overlap1_ref,
                                                   overlap2_ref,
                                                   plot_order)
        # Give some feedback
        logger.info("Best shift: {} covariance: {}".
                    format(best_trans, best_cov))
    else:
        best_trans = np.zeros(nr_dim, dtype=int)
        best_cov = np.nan
        logger.info("Best shift: {}. One of the neighbours is empty"
                    .format(best_trans))

    return list(best_trans), best_cov, contig_ind


def align_single_compress_pic(overlap1, overlap2, nr_peaks, nr_dim,
                              nr_slices,
                              plot_order):
    """Perform the alignment when using the 3D method "compress pic"

    Parameters:
    -----------

    overlap1: np.array
        Image that overlaps with overlap2
    overlap2: np.array
        Image that overlaps with overlap1
    nr_peaks: int   
        The n highest peaks from the PCM
        matrix that will be used to do crosscovariance
        with. A good number for 2D analysis is 8 peaks and
        good numbers for 3D with method ='compress pic' are
        6 or 9 peaks.
    nr_dim: int
        If 3, the code will assume three
        dimensional data for the tile, where z is the first
        dimension and y and x the second and third. For any
        other value 2-dimensional data is assumed.
        (default: 2)
    nr_slices: int
        Only applicable when running with 3D
        pictures and using 'compres pic' method. Determines
        the number of slices that are compressed together
        (compression in the z-direction).
    plot_order: np.array
        Numpy array, filled with ones. The order in wich subplots should be made
        if we want to plot overlap1 and 2


    Returns:
    --------

    best_trans: np.array
        1 by 3 array containing best found z, y and x translation.
    best_cov: float
        The covariance of the overlap after translation of the overlap by best_trans.
    """
    # Unraveled possible transistions contains for each substack
    # a list which contains for each of the 3 dimensions
    logger.info("Calculating pairwise alignment using compress pic "
                + "method.")
    best_trans_list = []
    best_cov_list = []
    unr_pos_transistions = []
    counter = 0
    # Change number of peaks, because we are dealing with 3 dimensions
    nr_peaks = int(np.rint(nr_peaks / 3))
    logger.debug("nr_peaks data type: {}".format(type(nr_peaks)))
    while counter < overlap1.shape[0]:
        com_overlap1 = []
        com_overlap2 = []
        for i in range(overlap1.ndim):
            com_overlap1.append(
                np.amax(overlap1[counter:counter + nr_slices, :, :]
                        , axis=i))
            com_overlap2.append(
                np.amax(overlap2[counter:counter + nr_slices, :, :]
                        , axis=i))

            # plt.figure('compressed image')
            # plt.imshow(com_overlap1[-1])
            # plt.show()
        unr_pos_transistions.append(calculate_pos_shifts(
            com_overlap1, com_overlap2, nr_peaks, nr_dim))
        counter += nr_slices

        # Do correlation over the found shifts:
        best_compr_trans = np.zeros((len(unr_pos_transistions[-1]), 2),
                                    dtype=int)
        best_compr_cov = np.zeros(len(unr_pos_transistions[-1]))
        # Test all transistions
        for com_dim in range(len(unr_pos_transistions[-1])):
            best_compr_trans[com_dim], best_compr_cov[
                com_dim] = find_best_trans(
                unr_pos_transistions[-1][com_dim],
                com_overlap1[com_dim], com_overlap2[com_dim],
                plot_order)
        logger.debug('best_compr_trans: {}'
                     .format(best_compr_trans))
        # Find the best translation based on which translation has the highest covariance
        # Sort the covariance
        cov_order = np.argsort(best_compr_cov)
        logger.debug('cov_order: {}'
                     .format(cov_order))
        # The translation in each direction is hidden in the 3 flattened pictures.
        # The posible z trasistions are at best_compr_trans[1,0] and best_compr_cov[2,0]
        # The posible y trasistions are at best_compr_trans[0,0] and best_compr_cov[2,1]
        # The posible x trasistions are at best_compr_trans[0,1] and best_compr_cov[1,1]
        # Based get the row indexes for the best translation in the z, y and x direction:
        z_ind = cov_order[max(np.nonzero(cov_order == 1)[0][0],
                              np.nonzero(cov_order == 2)[0][0])]
        logger.debug('z_ind index in cov_order options: {} {}'
                     .format(np.nonzero(cov_order == 1)[0][0],
                             np.nonzero(cov_order == 2)[0][0]))
        y_ind = cov_order[max(np.nonzero(cov_order == 0)[0][0],
                              np.nonzero(cov_order == 2)[0][0])]
        logger.debug('y_ind index in cov_order options: {} {}'
                     .format(np.nonzero(cov_order == 0)[0][0],
                             np.nonzero(cov_order == 2)[0][0]))
        x_ind = cov_order[max(np.nonzero(cov_order == 0)[0][0],
                              np.nonzero(cov_order == 1)[0][0])]
        logger.debug('x_ind index in cov_order options: {} {}'
                     .format(np.nonzero(cov_order == 0)[0][0],
                             np.nonzero(cov_order == 1)[0][0]))
        logger.debug('z_ind, y_ind, x_ind: {} {} {}'
                     .format(z_ind, y_ind, x_ind))
        best_trans = np.zeros(3, dtype=int)
        # Z translation
        best_trans[0] = best_compr_trans[z_ind, 0]
        best_cov = best_compr_cov[z_ind]
        logger.debug('z compr cov: {} best cov overall: {}'
                     .format(best_compr_cov[z_ind], best_cov))
        # Y translation
        if y_ind == 0:
            best_trans[1] = best_compr_trans[y_ind, 0]
            best_cov = best_cov + best_compr_cov[y_ind]
        elif y_ind == 2:
            best_trans[1] = best_compr_trans[y_ind, 1]
            best_cov = best_cov + best_compr_cov[y_ind]
        else:
            logger.warning(
                'y_ind has an invalid value, gonna raise an error')
            raise IndexError(
                'y_ind has an invalid value, it should be 0 or 2, it is currently {}'
                .format(y_ind))
        logger.debug('y compr cov: {} best cov overall: {}'
                     .format(best_compr_cov[y_ind], best_cov))
        # X translation
        best_trans[2] = best_compr_trans[x_ind, 1]
        best_cov = (best_cov + best_compr_cov[x_ind]) / 3.0
        logger.debug('x compr cov: {} best cov overall: {}'
                     .format(best_compr_cov[x_ind], best_cov))
        best_trans_list.append(best_trans)
        best_cov_list.append(best_cov)
    # Pick the best
    logger.debug('best_trans_list: {}'.format(best_trans_list))
    best_trans = best_trans_list[np.argmax(best_cov_list)]
    best_cov = np.nanmax(best_cov_list)

    return best_trans, best_cov


def determine_overlap(ind1, ind2, tile_1, tile_2, micData):
    """Determine the overlap between two neighbouring tiles

    Parameters:
    -----------

    ind1: int
        Index (flattened) of tile 1
    ind2: int
        Index (flattened) of tile 2
    tile_1: np.array
        Image 1
    tile_2: np.array
        Image 2
    micData: object
        MicroscopeData object containing coordinates

    Returns:
    --------

    overlap1: np.array
        Overlapping part of tile_1
    overlap2: np.array
        Overlapping part of tile_2
    plot_order: np.array
        Numpy array of ones. The shape of this array is
        used for plotting the overlaps in well fitting
        subplots.
    """
    if abs(ind1 - ind2) == 1:
        # Overlap on left or right
        logger.info(("Calculating overlap: right of tile {0} and "
                      + "left of tile {1}").format(ind1, ind2))
        logger.debug(
            "Ind: {0} , tile nr: {1} , x-coord: {2}"
            .format(ind1, micData.tile_set.flat[:][ind1],
                    micData.x_coords[micData.tile_set.flat[:][ind1]]))
        logger.debug(
            "Ind: {0} , tile nr: {1} , x-coord: {2}"
            .format(ind2, micData.tile_set.flat[:][ind2],
                    micData.x_coords[micData.tile_set.flat[:][ind2]]))
        # Calculate overlap indexes
        overlap_ind_x = int(
            micData.x_coords[micData.tile_set.flat[:][ind1]]
            - micData.x_coords[micData.tile_set.flat[:][ind2]])
        overlap_ind_y = int(
            micData.y_coords[micData.tile_set.flat[:][ind1]]
            - micData.y_coords[micData.tile_set.flat[:][ind2]])
        logger.debug("Overlap index, x: {} y: {}".
                     format(overlap_ind_x, overlap_ind_y))
        overlap1, overlap2 = ph.get_overlapping_region(tile_1, tile_2,
                                                       overlap_ind_x,
                                                       overlap_ind_y,
                                                       'left')
        # For subplotting the overlaps later:
        plot_order = np.ones((1, 3))
    else:
        # Overlap on top or bottom
        logger.info(("Calculating overlap: bottom of tile {0} and "
                     + "top of tile {1}").format(ind1, ind2))
        logger.debug(
            "Ind: {0} , tile nr: {1} , y-coord: {2}"
            .format(ind1, micData.tile_set.flat[:][ind1],
                    micData.y_coords[micData.tile_set.flat[:][ind1]]))
        logger.debug(
            "Ind: {0} , tile nr: {1} , y-coord: {2}"
            .format(ind2, micData.tile_set.flat[:][ind2],
                    micData.y_coords[micData.tile_set.flat[:][ind2]]))
        # Calculate overlap indexes
        overlap_ind_y = int(
            micData.y_coords[micData.tile_set.flat[:][ind1]]
            - micData.y_coords[micData.tile_set.flat[:][ind2]])
        overlap_ind_x = int(
            micData.x_coords[micData.tile_set.flat[:][ind1]]
            - micData.x_coords[micData.tile_set.flat[:][ind2]])
        logger.debug("Overlap index, y: {} x: {}".
                     format(overlap_ind_y, overlap_ind_x))
        overlap1, overlap2 = ph.get_overlapping_region(tile_1, tile_2,
                                                       overlap_ind_x,
                                                       overlap_ind_y,
                                                       'top')
        # For subplotting the overlaps later:
        plot_order = np.ones((3, 1))

    return (overlap1, overlap2, plot_order)


def calculate_pos_shifts(overlap1, overlap2, nr_peaks, nr_dim):
    """Calulate possible shifts, given two overlapping images

    Parameters:
    -----------

    overlap1: np.array
        Image that overlaps with overlap2.
    overlap2: np.array
        Image that overlaps with overlap1.
    nr_peaks: int
        The number of peaks from the PCM that will be used to calculate shifts
    nr_dim: int
        If 3, the code will assume three
        dimensional data for the tile, where z is the first
        dimension and y and x the second and third. For any
        other value 2-dimensional data is assumed.

    Returns:
    --------
    unr_pos_transistions: np.array
        Numpy array or list (list only when
        method == 'compress pic' and nr_dim == 3 )
        Numpy array numpy arrays of int, with each
        of the inner arrays containing the (z), x
        and y translation, if nr_dim is not 3 only
        x and y translation are given.
        If method == 'compress pic' and nr_dim == 3
        a list of 3 lists is returned. In each list
        the best translations for each compressed
        picture are given as numpy arrays of length
        2.

    """
    # Calculate phase correlation matrix of the overlap
    if nr_dim == 3:
        logger.info("Calculating posible shifts, method: {}"
                    .format(method))
        if method == 'use whole pic':
            r1 = ph.calculate_PCM(overlap1, overlap2)
            # inout.plot_3D(r1)
        elif method == 'calculate per layer':
            r1 = ph.calculate_PCM_method2(overlap1, overlap2)
            # inout.plot_3D(r1)
        elif method == 'compress pic':
            logger.debug('length compressed overlap list: {}'.format(
                len(overlap1)))
            r1_list = []
            for i in range(len(overlap1)):
                r1 = ph.calculate_PCM(overlap1[i], overlap2[i])
                r1_list.append(r1)
                # inout.display_tiles(r1_list, np.ones((1,3)))
    else:
        logger.info("Calculating posible shifts, in 2D")
        r1 = ph.calculate_PCM(overlap1, overlap2)
    # Get the first nr_peaks peaks
    if method == 'compress pic' and nr_dim == 3:
        unr_pos_transistions = [[], [], []]
        collect_zeros = []
        for i in range(len(r1_list)):
            cur_trans = np.argsort(r1_list[i].flat[:])[-nr_peaks:]
            unr_cur_trans = np.array(
                np.unravel_index(cur_trans, r1_list[i].shape)).T
            logger.debug("unr_cur_trans: {}".format(unr_cur_trans))

            # Calculate correct transition from the found peaks (calculation as in Skimage)
            # noinspection PyTypeChecker
            midpoints = np.array([np.fix(axis_size / 2) for axis_size in
                                  overlap1[i].shape])
            logger.debug('Midpoints: {}'.format(midpoints))
            for trans in unr_cur_trans:
                logger.debug('trans bigger than midpoints: {}'.format(
                    trans > midpoints))
                trans[trans > midpoints] -= np.array(overlap1[i].shape)[
                    trans > midpoints]
                trans[trans < (-1 * midpoints)] \
                    += np.array(overlap1[i].shape)[
                    trans < (-1 * midpoints)]
                unr_pos_transistions[i].append(trans)
                # If trans is not all zeros, multiply transistion1 by -1
                # to get the transistions the both ways
                if trans.any():
                    unr_pos_transistions[i].append(-1 * trans)
            # Add zero translation if not already there
            collect_zeros += [pos for pos in unr_pos_transistions[i] if
                              not (np.any(pos))]
            logger.debug("Collect_zeros: {}".format(collect_zeros))
            if not (len(collect_zeros)):
                unr_pos_transistions[i].append(
                    np.zeros((len(unr_pos_transistions[i][0])),
                             dtype=int))
                logger.debug("Added zero trans to transistion")
                logger.debug(
                    "Possible transistion after appending zeros: {}"
                    .format(unr_pos_transistions[i]))
    else:
        pos_transistions1 = np.argsort(r1.flat[:])[-nr_peaks:]

        # Unravel the found shifts
        unr_pos_transistions1 = np.array(
            np.unravel_index(pos_transistions1, r1.shape)).T
        logger.debug("pos_trans1: {}".format(unr_pos_transistions1))

        if method == 'calculate per layer' and nr_dim == 3:
            # Calculate correct transition from the found peaks (calculation as in Skimage)
            logger.debug('Overlap shape for midpoints: {}'.format(
                overlap1.shape[-2:]))
            # noinspection PyTypeChecker
            midpoints = np.array([np.fix(axis_size / 2) for axis_size in
                                  overlap1.shape[-2:]])
            logger.debug('Midpoints: {}'.format(midpoints))
            for trans in unr_pos_transistions1:
                logger.debug('trans bigger than midpoints: {}'.format(
                    trans[-2:] > midpoints))
                trans[-2:][trans[-2:] > midpoints] -= \
                np.array(overlap1.shape)[-2:][trans[-2:] > midpoints]
                trans[-2:][trans[-2:] < (-1 * midpoints)] \
                    += np.array(overlap1.shape)[-2:][
                    trans[-2:] < (-1 * midpoints)]
        else:
            # Calculate correct transition from the found peaks
            # (calculation as in Skimage)
            midpoints = np.array(
                [np.fix(axis_size / 2) for axis_size in overlap1.shape])
            logger.debug('Midpoints: {}'.format(midpoints))
            for trans in unr_pos_transistions1:
                logger.debug('trans bigger than midpoints: {}'.format(
                    trans > midpoints))
                trans[trans > midpoints] -= np.array(overlap1.shape)[
                    trans > midpoints]
                trans[trans < (-1 * midpoints)] \
                    += np.array(overlap1.shape)[
                    trans < (-1 * midpoints)]
        # Multiply transistion1 by -1 to get the transistions the both
        # ways
        inv_transistion = -1 * unr_pos_transistions1
        # And remove all zero transistions from the inverted, because
        # these are duplicates
        inv_transistion = inv_transistion[
            ~np.all(inv_transistion == 0, axis=1)]
        unr_pos_transistions = np.vstack(
            (unr_pos_transistions1, inv_transistion))
        # Add zero translation if not already there
        collect_zeros = [pos for pos in unr_pos_transistions if
                         not (np.any(pos))]
        if not (len(collect_zeros)):
            unr_pos_transistions = np.append(unr_pos_transistions,
                                             np.zeros((1, len(
                                                 unr_pos_transistions[
                                                     0])), dtype=int),
                                             axis=0)
            logger.debug("Added zero trans to transistion")

    return unr_pos_transistions


def find_best_trans(pos_transistions, overlap1, overlap2, plot_order):
    """Find the best translation using the cross covariance.

    Shift overlap according to translations and test the cov of
    the translated overlaps.

    Parameters:
    -----------

    pos_transistions: np.array
        2D numpy array. Array containing y,x-pairs denoting the possible translations.
    overlap1: np.array
        Image
    overlap2: np.array
        Image that overlaps with overlap1.
    plot_order: np.array
        The shape of this array denotes
        the order in wich subplots should be made
        if we want to plot overlap1 and 2.

    Returns:
    --------

    best_trans: np.array
        1 by 2 or 3 array containing best found (z), y and x translation.
    best_cov: float
        The covariance of the overlap after translation of the overlap by best_trans.
    """
    # Init best_trans and best_cov
    # This makes sure that if all correlations are below threshold,
    # we get no translation.
    logger.info("Finding best translation.")

    best_trans = np.zeros(overlap1.ndim, dtype=int)  # np.array([0,0])
    best_cov = 0.0

    logger.debug('best_trans at start {}, type: {}'.format(best_trans,
                                                    type(best_trans)))

    for trans in pos_transistions:
        if method == 'calculate per layer' and overlap1.ndim == 3:
            shifted_a, shifted_b = ph.calc_translated_pics_3D(
                trans[-2:],
                overlap1, overlap2)
        elif method == 'use whole pic' and overlap1.ndim == 3:
            shifted_a, shifted_b = ph.calc_translated_pics_3D(trans[-2:],
                                                              overlap1,
                                                              overlap2)
        else:
            shifted_a, shifted_b = ph.calc_translated_pics(trans,
                                                           overlap1,
                                                           overlap2)
        # Calculate xcov:
        if shifted_a.ndim == 2:
            cov, monocolor = ph.xcov_nd(shifted_a, shifted_b)
            logger.debug('Found a 2D picture to compare')
        else:
            cov_list = []
            monocolor_list = []
            for i in range(shifted_a.shape[0]):
                cov, monocolor = ph.xcov_nd(shifted_a[i, :, :],
                                            shifted_b[i, :, :])
                cov_list.append(cov)
                monocolor_list.append(monocolor)
            cov_list = np.array(cov_list)
            monocolor_list = np.array(monocolor_list)
            cov = np.mean(cov_list)
            monocolor = monocolor_list.all()
            logger.debug("cov {} monocolor {}".format(cov, monocolor))
        # Check for monocolor images, they do not provide usefull cov
        if monocolor:
            logger.debug(
                "Monocolor image found, covariance for these images is zero")
            cov = 0
        if (cov > best_cov):
            best_cov = cov
            best_trans = trans
            # Give some feedback
            # inout.display_tiles([shifted_a, shifted_b], plot_order, fig_nr = 8, block = True)
    # Check the result
    thr = 0.5
    if best_cov < thr:
        best_trans = np.zeros(overlap1.ndim,
                              dtype=int)
    # Line to call display overlap if we want to plot the overlap with the
    # best covariance
    if overlap1.ndim == 2:
        ph.display_overlap(overlap1, overlap2, best_trans[-2:],
                           best_cov, plot_order)
    else:
        ph.display_overlap(overlap1[0, :, :], overlap2[0, :, :],
                           best_trans[-2:], best_cov, plot_order)
    logger.debug('best_trans at end {}, type: {}'.format(best_trans,
                                                         type(
                                                             best_trans)))
    return best_trans, best_cov


def plot_overlaps(alignment, tiles, contig_tuples, micData):
    """Plot the pairwise overlaps

    

    Parameters:
    -----------

    alignment: dict
        Dictionary containing key 'P' with a flattened list of translations.
    tiles: list
        List of strings. Each string points to a tile in the hdf5 file.
    contig_tuples: list
        List of tuples denoting which tiles are contingent to each other.
    micData: object
        MicroscopeData object containing coordinates.


    Notes:
    ------
    Should be tested and made to work in 3D ?_?

    """
    logger.info("Trying to plot overlaps, plot will only show when "
                + "display_overlap in pairwisehelper.py is True and "
                + "matplotlib is imported in inout.py.")
    for i in range(len(contig_tuples)):
        # Pick 2 images to compare:
        ind1 = min(contig_tuples[i])
        ind2 = max(contig_tuples[i])
        logger.debug("Current indexes: {}, {}".format(ind1, ind2))
        if (tiles[ind1].any() and tiles[ind2].any()):
            # Determine overlap
            overlap1, overlap2, plot_order = determine_overlap(
                ind1, ind2,
                tiles, micData, None)
            trans = alignment['P'][i * 2: i * 2 + 2]
            logger.debug("Cur trans to be checked: {}".format(trans))
            ph.display_overlap(overlap1, overlap2, trans, None,
                               plot_order)



def align_single_pair_npy(contig_tuple,filtered_files_list,micData, 
                          nr_peaks=8):
    
    """
    Determine the ideal alignment between two neighbouring tiles
    It is a modification of the align_single_pair
    function that will run in parallel using .npy image arrays.
    The functions runs only in 2D.
    
    Parameters:
    -----------

    contig_tuple: tuple
        Tuple containing two tile indexes denoting these
        tiles are contingent to each other.
    filtered_files_list: list
        List containing the paths to the files to porocess
    micData: object
        MicroscopeData object. Should contain coordinates of
        the tile corners as taken from the microscope.
        These coordinates are used to dtermine the overlap
        between a tile pair.
    nr_peaks: int
        The n highest peaks from the PCM
        matrix that will be used to do crosscovariance
        with. A good number for 2D analysis is 8 peaks and
        good numbers for 3D with method ='compress pic' are
        6 or 9 peaks.

    Returns:
    --------

    best_trans: np.array
        1 by 2 or 3 array containing best found (z), y and x translation.
    best_cov: float
        The covariance of the overlap after translation of the overlap by best_trans.
    contig_ind: int
        The index of the used tile pair in
        contig_tuples. This is necessary to return when
        running on multiple processors/cores. More
        precisely: the index of the tuple in contig_tuples
        containing the indexes of the tiles that should be
        aligned.
    """
    
    # This function run only for 2D so I can hard code the nr_dim
    nr_dim = 2
    
    # The contig_tuple contains the indexes derived from the tile_set

    ind1 = min(contig_tuple)
    ind2 = max(contig_tuple)
    
    logger.info("Calculating pairwise alignment for indexes: {} and {}"
                .format(ind1, ind2))
    
    # Load the images
    # Identify the tile positioned in the original image set
    tile_1_pos = micData.tile_set.flat[:][ind1]
    tile_2_pos = micData.tile_set.flat[:][ind2]
    
    
    
    tile_1_fpath = [fpath for fpath in filtered_files_list if 'pos_'+str(tile_1_pos)+'.' in fpath]
    tile_2_fpath = [fpath for fpath in filtered_files_list if 'pos_'+str(tile_2_pos)+'.' in fpath]
    if tile_1_fpath:
        tile_1 = np.load(tile_1_fpath[0])
        tile_1 =img_as_float(tile_1)
    else:
        tile_1 = np.array([],dtype=np.float64)

    if tile_2_fpath:
        tile_2 = np.load(tile_2_fpath[0])
        tile_1 =img_as_float(tile_1)
    else:
        tile_2 = np.array([],dtype=np.float64)
    
        
    if (tile_1.size and tile_2.size):
#         Determine overlap
        overlap1, overlap2, plot_order = determine_overlap(ind1,
                                                           ind2, tile_1,
                                                           tile_2,
                                                           micData)
    
        logger.debug("Shape of overlap 1 and 2: {} {}"
                     .format(overlap1.shape, overlap2.shape))
    
        unr_pos_transistions = calculate_pos_shifts(
                overlap1, overlap2, nr_peaks, nr_dim)

        # Do correlation over the found shifts:
        best_trans, best_cov = find_best_trans(unr_pos_transistions,
                                               overlap1, overlap2,
                                               plot_order)

         
        # Give some feedback
        logger.info("Best shift: {} covariance: {}".format(best_trans,
                                                           best_cov))
        logger.debug(
            "Best shift type: {} {} covariance type: {} contig_ind type: {}"
            .format(type(best_trans), type(best_trans[0]),
                    type(best_cov), type(contig_tuple)))
        
    else:
        # One of the tiles is empty
        best_trans = np.zeros(nr_dim, dtype=int)
        best_cov = np.nan
        
        logger.info(
            "Best shift: {}. One of the neighbours is empty".format(
                best_trans))
    
    return np.array(best_trans,dtype='int16'), best_cov, contig_tuple