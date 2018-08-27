import h5py
import math
import numpy as np
import logging

from . import tilejoining
from . import inout

""" Functions to create necessary structures and fill them in the hfd5
file.

Functions:
create_structures_hdf5_files
fill_hdf5_files
divide_final_image
fill_hdf5_MPI

"""
logger = logging.getLogger(__name__)


########################### HDF5 file creation ##########################
def create_structures_hdf5_files(stitching_file, joining, nr_pixels,
                                 z_size, hyb_nr, gene,
                                 ubyte = True, blend = 'non linear',
                                 reg_name = ''):
    """Takes an HDF5 file handle and creates the necessary structures.

    Creates groups and data sets, when the groups or data sets already
    exists, they are kept as they are, as long as the data sets have
    the right size and data type. Incompatible data sets will be
    overwritten.
    Stitching file has the following structure:

    __________________Groups____________|___Data sets____

    gene1:
          StitchedImage:
                                        final_image
                                        final_image_ubyte
                                        blending_mask
                                        temp_mask
          StitchingTempTiles:
                         blended_tiles:
                                        0
                                        1
                                        .
                                        .
                                        n
                         ubytes:
                                        0
                                        1
                                        .
                                        .
                                        n
                         temp_masks:
                                        0
                                        1
                                        .
                                        .
                                        n
   gene2: ...



    Parameters:
    -----------

    stitching_file: pointer
        HDF5 file handle. The file where the stitched images will be saved.
    joining: dict
        Dictionary containing keys 'corner_list  and 'final_image_shape'. 
        Corner_list is a list of list, each list is a pair of an image number
        (int) and it's coordinates (numpy array containing floats).
        Final_image_shape is a tuple of size 2 or 3 depending on the number of 
        dimensions and contains ints.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_size: int 
        The number of layers in one tile (size of the z-axis). 
        Should be 1 or None if the tile is 2D.
    hyb_nr: int 
        The number of the hybridization we are stitching. 
        This will be used to place the data in the right group in stitching_file.
    gene: str
        The name of the gene we are stitching.This will be used to place the
        data in the right group in stitching_file.
    ubyte: bool 
        We need to create a data set for the ubyte image when True.
        Otherwise we only need to create a data set for the high resolution image.
    blend: str 
        When 'non linear' or 'linear', blending will be applied, so we will 
        need to create the structures necessary for saving the blended tiles. When
        it has another value or is None no blending at all will be applied, 
        so we can skip this. This variable also determines to return value of
        linear_blending.
    reg_name: str 
        A name that will be used to create a second StitchedImage group for 
        an image that is registered to be aligned with the other hybridizations. 
        warp_name will be attached to the end of the group name, which is
        StitchedImage.
    
    Returns:
    --------

    stitched_group: pointer
        HDF5 reference to the group where the final will be.
    temp_group: pointer
        HDF5 reference to the group where the blended tiles will be saved.
    linear_blending: bool
        When True later blending should be linear and when False, blending
        should be non-linear.
    ubyte: bool
        The ubyte image should be saved when True. Otherwise only the high
        resolution image should be saved.
    blend: str 
        When 'non linear' or 'linear', blending should be applied. When
        it has another value or is None no blending at all will be applied.


    """
    logger.info("Generating stitching file structures.")
    # Create a group for the stitched images in the stitching file
    #stitching_file.require_group('Hybridization' + str(hyb_nr))
    stitching_file.require_group(gene)
    stitched_group = stitching_file[gene].require_group('StitchedImage' + reg_name)
    # Create the final image in this file
    try:
        final_image = stitched_group.require_dataset('final_image',
                                                joining['final_image_shape'],
                                                dtype = np.float64)
    except TypeError as err:
        logger.info("Incompatible 'final_image' data set already existed, deleting old dataset.\n {}"
                    .format(err))
        del stitched_group['final_image']
        inout.free_hdf5_space(stitching_file)
        final_image = stitched_group.require_dataset('final_image',
                                                joining['final_image_shape'],
                                                dtype = np.float64)
    # Create a low resolution image
    if ubyte:
        try:
            final_image_ubyte = stitched_group.require_dataset('final_image_ubyte',
                                                       joining['final_image_shape'],
                                                       dtype=np.uint8)
        except TypeError as err:
            logger.info("Incompatible 'final_image_ubyte' data set already existed, deleting old dataset.\n {}"
                        .format(err))
            del stitched_group['final_image_ubyte']
            inout.free_hdf5_space(stitching_file)
            final_image = stitched_group.require_dataset('final_image_ubyte',
                                                       joining['final_image_shape'],
                                                       dtype=np.uint8)

    # If blending is required initialize the blending mask in the
    # hdf5 file
    if blend is not None:
        # For the blending masks use only the last 2 dimensions of final
        # image shape, because also when working in 3D the masks can be
        # 2D as there is the same shift in x and y direction for the
        # whole stack.
        try:
            blending_mask = stitched_group.require_dataset('blending_mask',
                                                   joining['final_image_shape'][-2:],
                                                   dtype = np.float64)
        except TypeError as err:
            logger.info("Incompatible 'blending_mask' data set already existed, deleting old dataset.\n {}"
                        .format(err))
            del stitched_group['blending_mask']
            inout.free_hdf5_space(stitching_file)
            final_image = stitched_group.require_dataset('blending_mask',
                                                   joining['final_image_shape'][-2:],
                                                   dtype = np.float64)
        try:
            temp_mask = stitched_group.require_dataset('temp_mask',
                                               joining['final_image_shape'][-2:],
                                               dtype = np.float64)
        except TypeError as err:
            logger.info("Incompatible 'temp_mask' data set already existed, deleting old dataset.\n {}"
                        .format(err))
            del stitched_group['temp_mask']
            inout.free_hdf5_space(stitching_file)
            final_image = stitched_group.require_dataset('temp_mask',
                                               joining['final_image_shape'][-2:],
                                               dtype = np.float64)


    # Check type of blending
    if blend == 'non linear':
        linear_blending = False
    elif blend == 'linear':
        linear_blending = True
    else:
        linear_blending = False
        logger.warning("Blend not defined correctly, \
                                using non-linear blending, \
                                blend is: {}".format(blend))

    # Generate the tmp_file structure
    temp_group = stitching_file[gene].require_group('StitchingTempTiles' + reg_name)
    blended_tiles = temp_group.require_group('blended_tiles')
    ubytes = temp_group.require_group('ubytes')
    temporary_masks = temp_group.require_group('temp_masks')

    for idx in range(len(joining['corner_list'])):
            if z_size == 1 or z_size is None:
                try:
                    blended_tiles.require_dataset(str(idx), (nr_pixels, nr_pixels), np.float64)
                except TypeError as err:
                    logger.debug("Incompatible data set for blended tile {}, deleting old dataset.\n {}"
                                .format(idx, err))
                    del blended_tiles[str(idx)]
                    inout.free_hdf5_space(stitching_file)
                    blended_tiles.require_dataset(str(idx), (nr_pixels, nr_pixels), np.float64)
                try:
                    ubytes.require_dataset(str(idx), (nr_pixels, nr_pixels), np.uint8)
                except TypeError as err:
                    logger.debug("Incompatible data set for ubyte tile {}, deleting old dataset.\n {}"
                                .format(idx, err))
                    del ubytes[str(idx)]
                    inout.free_hdf5_space(stitching_file)
                    ubytes.require_dataset(str(idx), (nr_pixels, nr_pixels), np.uint8)
            else:
                try:
                    blended_tiles.require_dataset(str(idx), (z_size, nr_pixels, nr_pixels), np.float64)
                except TypeError as err:
                    logger.debug("Incompatible data set for blended tile {}, deleting old dataset.\n {}"
                                .format(idx, err))
                    del blended_tiles[str(idx)]
                    inout.free_hdf5_space(stitching_file)
                    blended_tiles.require_dataset(str(idx), (z_size, nr_pixels, nr_pixels), np.float64)
                try:
                    ubytes.require_dataset(str(idx), (z_size, nr_pixels, nr_pixels), np.uint8)
                except TypeError as err:
                    logger.debug("Incompatible data set for ubyte tile {}, deleting old dataset.\n {}"
                                .format(idx, err))
                    del ubytes[str(idx)]
                    inout.free_hdf5_space(stitching_file)
                    ubytes.require_dataset(str(idx), (z_size, nr_pixels, nr_pixels), np.uint8)

            try:
                temporary_masks.require_dataset(str(idx), (nr_pixels, nr_pixels), np.float64)
            except TypeError as err:
                logger.debug("Incompatible data set for temporary masks tile {}, deleting old dataset.\n {}"
                                .format(idx, err))
                del temporary_masks[str(idx)]
                inout.free_hdf5_space(stitching_file)
                temporary_masks.require_dataset(str(idx), (nr_pixels, nr_pixels), np.float64)
    if False:
        logger.info("Flushing hdf5 file to clean up after delete operations")
        before_flush = stitching_file.id.get_filesize()
        stitching_file.flush()
        after_flush = stitching_file.id.get_filesize()
        logger.debug("Size in bytes before flush: {} after flush: {} space freed: {}".format(before_flush, after_flush, before_flush - after_flush))
    return stitched_group, temp_group, linear_blending, ubyte, blend


def fill_hdf5_files(im_file, joining, nr_pixels, ubyte, blend):
    """
    Fill the data sets in the hdf5 file with zeros or random data.

    Parameters:
    -----------

    im_file: pointer
        HDF5 reference. Reference to the StitchedImage
        group or any group or file that contains final_image
        and optionally final_image_ubyte, blending_mask and
        temp_mask.
    joining: dictionary 
        Containing the key final_image_shape.
        Final_image_shape is a tuple of size 2 or 3 depending on
        the number of dimensions and contains ints.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    ubyte: bool
        The ubyte image will be filled when True.
        Otherwise only the high resolution image will be saved.
    blend: str 
        When blend is not None, blending_mask and temp_mask will be filled.
    """
    logger.info("Filling stitching file data sets.")
    # Fill the image arrays with zeros, this works for 3D
    im_file['final_image'][:]= np.zeros(joining['final_image_shape'])
    if ubyte:
        im_file['final_image_ubyte'][:] = np.empty(joining['final_image_shape'],
                                           dtype=np.uint8)
    if blend is not None:
        im_file['blending_mask'][:] = np.zeros(joining['final_image_shape'][-2:])
        tilejoining.make_mask(joining, nr_pixels, im_file['blending_mask'])
        im_file['temp_mask'][:] = np.zeros(joining['final_image_shape'][-2:])


def divide_final_image(im_shape, nr_blocks):
    """
    Divide the image in rows if it is too large to be filled at once.

    The parallel HDF5 implementation cannot handle images larger
    than 2GB. Therefore this function can divide the rows of the image
    over nr_blocks to be filled by separate cores.

    Parameters:
    -----------

    im_shape: tuple
        Shape (y, x) of the image to be filled.
    nr_blocks: int
        The number of block to divide the image in,
        typically this will be the number of available cores.

    Returns:
    --------

    y_corners: list
        List of numpy ints (numpy.int64). The y-corners of all the blocks.
    """

    logger.debug("nr_blocks {}".format(nr_blocks))
    logger.debug("im_shape {}".format(im_shape[-2]))
    y_corners = np.linspace(0, im_shape[-2], num = nr_blocks,
                        endpoint = False, dtype = int)
    logger.debug("y_corners {}".format(y_corners))
    logger.debug('Types: y_corners: {} corner element: {}'
                    .format(type(y_corners), type(y_corners[0])))
    return y_corners


def fill_hdf5_MPI(y_corners, ind, im_shape, im_file, ubyte, blend):
    """Fill the data sets in the hdf5 file with zeros, block by block.

    The parallel HDF5 implementation cannot handle images larger
    than 2GB. Therefore this function only fills the part of the
    image from the indexing value found in y_corners[ind] up to
    y_corners[ind + 1].

    Parameters:
    -----------

    y_corners: list
        List of numpy ints (numpy.int64). The y-corners of all the blocks.
    ind : int 
        The index of the corner in y_corners
    im_shape: tuple
        Shape of the final image.
    im_file: pointer 
        HDF5 file handle or group reference. The file or
        group with the datasets that need to be filled.
    ubyte: bool
        Ubyte image is filled if True
    blend: str 
        If this is None datasets needed for blending
        are left empty, for all other values they are filled.
    """
    logger.info("Filling stitching file data sets in MPI mode.")
    # Calculate the range to be filled with zeros
    y_min = y_corners[ind]
    if (ind + 1) < len(y_corners):
        y_max = y_corners[ind + 1]
    else:
        y_max = im_shape[-2]

    logger.debug("index: {} y_min: {} y_max: {} im_shape[0]: {}"
                .format(ind, y_min, y_max, im_shape[-2]))

    # Fill with zeros:
    if len(im_shape) == 3:
        im_file['final_image'][:, y_min:y_max,:] = 0.0
        if ubyte:
            im_file['final_image_ubyte'][:, y_min:y_max,:] = 0
    else:
        im_file['final_image'][y_min:y_max,:] = 0.0
        if ubyte:
            im_file['final_image_ubyte'][y_min:y_max,:] = 0
    if blend is not None:
        im_file['blending_mask'][y_min:y_max,:] = 0.0
        im_file['temp_mask'][y_min:y_max,:] = 0.0

    #DO NOT FORGET to run tilejoining.make_mask after filling the arrays!!!!


########################### HDF5 file creation ##########################
def create_structures_hdf5_stitched_ref_gene_file_npy(stitching_file, joining, nr_pixels,
                                 reference_gene, blend = 'non linear'):
    """Takes an HDF5 file handle and creates the necessary structures.

    Modification of create_structures_hdf5_files to work with .npy list of
    files

    Creates groups and data sets, when the groups or data sets already
    exists, they are kept as they are, as long as the data sets have
    the right size and data type. Incompatible data sets will be
    overwritten.
    Stitching file has the following structure:

    __________________Groups____________|___Data sets____

    gene_stitched:
          StitchedImage:
                                        final_image
                                        blending_mask


    Parameters:
    -----------

    stitching_file: pointer
        HDF5 file handle. The file where the stitched images will be saved.
    joining: dict 
        Dictionary containing keys 'corner_list  and 'final_image_shape'. 
        Corner_list is a list of list, each list is a pair of an image number
        (int) and it's coordinates (numpy array containing floats).
        Final_image_shape is a tuple of size 2 or 3
        depending on the number of dimensions and contains ints.
    nr_pixels: int 
        Height and length of the tile in pixels, tile is assumed to be square.
    reference_gene: str 
        The name of the gene we are stitching.This will be used to place the
        data in the right group in stitching_file.
     blend: str
        When 'non linear' or 'linear',blending will be applied,
        so we will need to create the structures
        necessary for saving the blended tiles. When
        it has another value or is None no blending at
        all will be applied, so we can skip this.
        This variable also determines to return value of
        linear_blending.
    
    Returns:
    --------

    stitched_group: pointer
        HDF5 reference to the group where the final will be.
    linear_blending: bool
        When True later blending should be linear and when False, blending
        should be non-linear.
    blend: str 
        When 'non linear' or 'linear', blending should be applied. When
        it has another value or is None no blending at
        all will be applied.


    """
    logger.info("Generating stitching file structures.")
    # Create a group for the stitched images in the stitching file
    
    stitching_file.require_group(reference_gene)
    stitched_group = stitching_file[reference_gene].require_group('StitchedImage')
    # Create the final image in this file
    try:
        final_image = stitched_group.require_dataset('final_image',
                                                joining['final_image_shape'],
                                                dtype = np.float64)
    except TypeError as err:
        logger.info("Incompatible 'final_image' data set already existed, deleting old dataset.\n {}"
                    .format(err))
        del stitched_group['final_image']
        inout.free_hdf5_space(stitching_file)
        final_image = stitched_group.require_dataset('final_image',
                                                joining['final_image_shape'],
                                                dtype = np.float64)
    

    # If blending is required initialize the blending mask in the
    # hdf5 file
    if blend is not None:
        # For the blending masks use only the last 2 dimensions of final
        # image shape, because also when working in 3D the masks can be
        # 2D as there is the same shift in x and y direction for the
        # whole stack.
        try:
            blending_mask = stitched_group.require_dataset('blending_mask',
                                                   joining['final_image_shape'][-2:],
                                                   dtype = np.float64)
        except TypeError as err:
            logger.info("Incompatible 'blending_mask' data set already existed, deleting old dataset.\n {}"
                        .format(err))
            del stitched_group['blending_mask']
            inout.free_hdf5_space(stitching_file)
            final_image = stitched_group.require_dataset('blending_mask',
                                                   joining['final_image_shape'][-2:],
                                                   dtype = np.float64)

    # Check type of blending
    if blend == 'non linear':
        linear_blending = False
    elif blend == 'linear':
        linear_blending = True
    else:
        linear_blending = False
        logger.warning("Blend not defined correctly, \
                                using non-linear blending, \
                                blend is: {}".format(blend))

    
    if False:
        logger.info("Flushing hdf5 file to clean up after delete operations")
        before_flush = stitching_file.id.get_filesize()
        stitching_file.flush()
        after_flush = stitching_file.id.get_filesize()
        logger.debug("Size in bytes before flush: {} after flush: {} space freed: {}".format(before_flush, after_flush, before_flush - after_flush))
    return stitched_group, linear_blending, blend