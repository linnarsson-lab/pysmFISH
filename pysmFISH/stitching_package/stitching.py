
"""Find or apply coordinates to stitch an image
"""

#import matplotlib.pyplot as plt
#plt_available = True
plt_available = False
import sklearn.feature_extraction.image as sklim
import numpy as np
import skimage.transform as smtf
import h5py
import logging
import time
import os

# Own imports
from . import inout
from . import pairwisesingle as ps
from .MicroscopeData import MicroscopeData
from .GlobalOptimization import GlobalOptimization
from . import tilejoining

from .. import utils

# Logger
logger = logging.getLogger(__name__)

#################### Initial stitching functions #######################
def get_pairwise_input(ImageProperties,folder, tile_file, hyb_nr, gene = 'Nuclei',
                        pre_proc_level = 'FilteredData',
                        est_overlap = 0.1, y_flip = False, nr_dim = 2):
    """Get the information necessary to do the pairwise allignment

    Find the pairwise pars for an unknown stitching.
    Works best with a folder containing
    image with nuclei (DAPI staining)

    Parameters:
    -----------

    folder: str
        String representing the path of the folder containing
        the tile file and the yaml metadata file. Needs a
        trailing slash ('/').
    tile_file: pointer
        HDF5 file handle. Reference to the opened file containing the tiles.
    hyb_nr: int
        The number of the hybridization we are going to
        stitch. This will be used to navigate tile_file and find
        the correct tiles.
    gene: str
        The name of the gene we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Nuclei')
    pre_proc_level: str
        The name of the pre processing group of
        the tiles we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Filtered')
    est_overlap: float
        The fraction of two neighbours that should
        overlap, this is used to estimate the shape of the
        tile set and then overwritten by the actual average
        overlap according to the microscope coordinates.
        (default: 0.1)
    y_flip: bool
        The y_flip variable is designed for the cases where the
        microscope sequence is inverted in the y-direction. When
        set to True the y-coordinates will also be inverted
        before determining the tile set. (Default: False)
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (Default: 2)
    
    Returns:
    --------

    tiles: list
        List of references to the the tiles in the hdf5 file tile_file.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of the z-axis). Is 1 when nr_dim is not 3.
    micData: object
        MicroscopeData object. Contains coordinates of the tile corners as taken from the microscope.
    """
    logger.info("Getting files from folder: {}".format(folder))

    # Load the information from the metadata file.
    ExperimentInfos, ImageProperties, HybridizationsInfos, \
    Converted_Positions, MicroscopeParameters = \
        utils.experimental_metadata_parser(folder)
    # Get coordinate data for this hybridization
    coord_data = Converted_Positions['Hybridization' + str(hyb_nr)]

    # Read the number of pixels, z-count and pixel size from the yaml
    # file.
    try:
        nr_pixels       = ImageProperties['HybImageSize']['rows']
    except KeyError as err:
        logger.info(("Number of pixels not found in experimental "
                     + "metadata file.\nPlease add "
                     + "the number of pixels in an image "
                     + "to the experimental "
                     + "metadata file under ImageProperties "
                     + "--> HybImageSize --> rows.\n"
                     + "KeyError: {}").format(err))
        raise

    if nr_dim == 2:
        z_count = 1
    else:
        try:
            z_count = ImageProperties['HybImageSize']['zcount']
        except KeyError as err:
            logger.info(("Number of pixels not found in experimental "
                         + "metadata file.\nPlease add "
                         + "the number of slices in the z-stack "
                         + "to the experimental "
                         + "metadata file under ImageProperties "
                         + "--> HybImageSize --> zcount.\n"
                         + "KeyError: {}")
                         .format(err))
            raise

    try:
        pixel_size = ImageProperties['PixelSize']
    except KeyError as err:
        logger.info(("ImageProperties['PixelSize'] not found in "
                    + "experimental metadata file.\nPlease add the "
                    + "size of a pixel in um in the experimental "
                    + "metadata file under ImageProperties "
                    + "--> PixelSize.\nKeyError: {}").format(err))
        raise
    # Estimate the overlap in pixels with the overlap that the user
    # provided, default is 10%
    est_x_tol = nr_pixels * (1 - est_overlap)
    logger.info("Estimating overlap at {}%, that is {} pixels"
                .format(est_overlap * 100, est_x_tol))
    logger.debug("Number of pixels: {}".format(nr_pixels))
    logger.debug("Number of slices in z-stack: {}".format(z_count))

    # Organize the microscope data and determine tile set
    micData = MicroscopeData(coord_data, y_flip, nr_dim)
    micData.normalize_coords(pixel_size)
    micData.make_tile_set(est_x_tol, nr_pixels = nr_pixels)

    # Make a list of image numbers, matching with the numbers in the
    # image files
    flat_tile_set = micData.tile_set.flat[:]

    image_list = [micData.tile_nr[ind] if ind >= 0 else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    logger.info("Getting references for: {}".format(image_list))

    # Make a list of the image names
    tiles = inout.get_image_names(tile_file, image_list = image_list,
                                    hyb_nr = hyb_nr, gene = gene,
                                    pre_proc_level = pre_proc_level)

    logger.info("Size tiles: {} Number of pixels: {} z count: {}"
                .format(len(tiles), nr_pixels, z_count))

    # Produce an undirected graph of the tiles, tiles that are
    # neighbours to each other are connected in this graph.
    # noinspection PyPep8Naming
    C = np.asarray(sklim.grid_to_graph(*micData.tile_set.shape).todense())
    np.fill_diagonal(C, 0)
    # noinspection PyPep8Naming
    C = np.triu( C )
    # Extract the neighbour pairs from the graph
    contig_tuples =list(zip( *np.where( C ) ))
    logger.info(("Length contingency tuples: {} \n"
                + "Contingency tuples: {}")
                 .format(len(contig_tuples), contig_tuples))

    # Plotting tiles:
    #inout.display_tiles(tiles, micData.tile_set, fig_nr = 2, block = False)
    #plt.show(block = True)
    return (tiles, contig_tuples, nr_pixels, z_count, micData)

def get_pairwise_alignments(tiles, tile_file, contig_tuples,
                                        micData, nr_peaks = 8,
                                        nr_slices = None,
                                        nr_dim = 2):
    """Calculate the pairwise transition

    Calculates pairwise transition for each neighbouring pair of
    tiles. This functions is only used in the single core version of the
    code, not when using MPI.

    Parameters:
    -----------

    tiles: list
        List of references to the the tiles in the hdf5 file tile_file.
    tile_file: pointer
        HDF5 file handle. Reference to the opened file containing the tiles.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    micData: object
        MicroscopeData object. Containing coordinates of
        the tile corners as taken from the microscope.
    nr_peaks: int
        Number of peaks to be extracted from the PCM (Default: 8)
    nr_slices: int 
        Only applicable when running with 3D
        pictures and using 'compres pic' method in
        pairwisesingle.py. Determines the number of slices
        that are compressed together (compression in the
        z-direction). If None, all the slices are compressed
         together. (Default: None)
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (default: 2)

    Returns:
    --------

    : dict
        Contains key 'P' with a 1D numpy array
        containing pairwise alignment y and x coordinates
        (and z-coordinates when applicable) for each
        neighbouring pair of tiles, array will be
        2 * len(contig_typles) for 2D data
        or 3 * len(contig_typles) for 3D data.
        Also contains key 'covs' with a 1D numpy array
        containing covariance for each pairwise alignment in
        'P', 'covs' will be len(contig_typles).
    """
    logger.info("Getting pairwise alignments...")
    P = np.empty((len(contig_tuples), nr_dim), dtype = int)
    covs = np.empty(len(contig_tuples))

    for i in range(len(contig_tuples)):
        # noinspection PyPep8Naming
        P_single, cov, contig_index = ps.align_single_pair(tiles, tile_file,
                                        contig_tuples, i,
                                        micData, nr_peaks,
                                        nr_slices = nr_slices,
                                        nr_dim = nr_dim)
        P[contig_index,:] = P_single
        covs[contig_index] = cov

    logger.info("Raw P: {}".format(P))
    #Flatten P
    P = np.array(P).flat[:]
    logger.info("flat P: {}".format(P))
    return {'P': P, 'covs': covs}

############################# Apply ####################################
def get_place_tile_input_apply(folder, tile_file, hyb_nr, data_name,
                                gene = 'Nuclei',
                                pre_proc_level = 'Filtered',
                                nr_dim = 2, check_pairwise = False):
    """Get the data needed to apply stitching to another gene

    Parameters:
    -----------

    folder: str
        String representing the path of the folder containing
        the tile file, the stitching data file the yaml metadata
        file. Needs a trailing slash ('/').
    tile_file: pointer
        HDF5 file handle. Reference to the opened file
        containing the tiles.
    hyb_nr: int
        The number of the hybridization we are going to
        stitch. This will be used to navigate tile_file and find
        the correct tiles.
    data_name: str
        Name of the file containing the pickled stitching data.
    gene: str
        The name of the gene we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Nuclei')
    pre_proc_level: str
        The name of the pre processing group of
        the tiles we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Filtered')
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (Default: 2)
    check_pairwise: bool
        If True the contig_tuples array is assumed
        to be in the pickled data file and will be returned.
        (Default: False)

    Returns:
    --------

    joining: dict
        Taken from the stitching data file.
        Contains keys corner_list and final_image_shape.
        Corner_list is a list of list, each list is a pair
        of an image number (int) and it's coordinates (numpy
        array containing floats).
        Final_image_shape is a tuple of size 2 or 3
        depending on the numer of dimensions and contains
        ints.
    tiles: list
        List of references to the the tiles in the hdf5 file tile_file.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of
        the z-axis). Is 1 when nr_dim is not 3.
    micData: object
        MicroscopeData object. Taken from the pickled
        stitching data.
        Contains coordinates of the tile corners as taken
        from the microscope.
    contig_tuples: list
        Only returned if check_pairwise == True.
        list of tuples. Taken from the pickled
        stitching data. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    """
    logger.info("Getting data to apply stitching from file...")

    # Load image list and old joining data
    stitching_coord_dict = inout.load_stitching_coord(folder + data_name)
    # noinspection PyPep8Naming
    micData = stitching_coord_dict['micData']
    joining = stitching_coord_dict['joining']
    logger.info("Joining object and image list loaded from file")

    # Make a list of image numbers
    flat_tile_set = micData.tile_set.flat[:]
    image_list = [micData.tile_nr[ind] if ind >= 0 else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    logger.info("Tile set size: {}".format(micData.tile_set.shape))
    logger.info("Placing folowing image references in tiles: {}"
                .format(image_list))

    # Make a list of the tile references
    tiles = inout.get_image_names(tile_file, image_list = image_list,
                                hyb_nr = hyb_nr, gene = gene, pre_proc_level = pre_proc_level)

    # Load the data from the metadata file
    ExperimentInfos, ImageProperties, HybridizationsInfos, \
    Converted_Positions, MicroscopeParameters = \
        utils.experimental_metadata_parser(folder)

    # Read the number of pixels and z-count from the yaml
    # file.
    try:
        nr_pixels = ImageProperties['HybImageSize']['rows']
    except KeyError as err:
        logger.info(("Number of pixels not found in experimental "
                     + "metadata file.\nPlease add "
                     + "the number of pixels in an image "
                     + "to the experimental "
                     + "metadata file under ImageProperties "
                     + "--> HybImageSize --> rows.\n"
                     + "KeyError: {}").format(err))
        raise

    if nr_dim == 2:
        z_count = 1
    else:
        try:
            z_count = ImageProperties['HybImageSize']['zcount']
        except KeyError as err:
            logger.info(
                ("Number of pixels not found in experimental "
                 + "metadata file.\nPlease add "
                 + "the number of slices in the z-stack "
                 + "to the experimental "
                 + "metadata file under ImageProperties "
                 + "--> HybImageSize --> zcount.\n"
                 + "KeyError: {}")
                .format(err))
            raise

    logger.info("Size tiles: {} Number of pixels: {} z count: {}"
                .format(len( tiles), nr_pixels, z_count))

    # Check pairwise overlap in signal
    if check_pairwise:
        contig_tuples   = stitching_coord_dict['contig_tuples']
        logger.info(
            "Length contingency tuples: {} Contingency tuples: {}"
            .format(len(contig_tuples), contig_tuples))
        return (joining, tiles, nr_pixels, z_count, micData,
                                            contig_tuples)
    else:
        return (joining, tiles, nr_pixels, z_count, micData)


############################# Refine ###################################
def get_refine_pairwise_input(folder, tile_file, hyb_nr, data_name,
                                gene = 'Nuclei', pre_proc_level = 'Filtered',
                                nr_dim = 2):
    """Get the data needed to refine stitching with another gene

    Parameters:
    -----------

    folder: str
        String representing the path of the folder containing
        the tile file, the stitching data file the yaml metadata
        file. Needs a trailing slash ('/').
    tile_file: pointer
        HDF5 file handle. Reference to the opened file
        containing the tiles.
    hyb_nr: int
        The number of the hybridization we are going to
        stitch. This will be used to navigate tile_file and find
        the correct tiles.
    data_name: str
        Name of the file containing the pickled stitching data.
    gene: str
        The name of the gene we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Nuclei')
    pre_proc_level: str
        The name of the pre processing group of
        the tiles we are going to stitch.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Filtered')
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (Default: 2)

    Returns:
    --------

    tiles: list
        List of references to the the tiles in the hdf5 file tile_file.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    nr_pixels: int
        Height and length of the tile in pixels, tile
        is assumed to be square.
    z_count: int
        The number of layers in one tile (size of the z-axis). Is 1 when nr_dim is not 3.
    micData: object
        MicroscopeData object. Contains coordinates of
        the tile corners as taken from the microscope.
    """
    logger.info("Aplying stitching from file")

    # Load image list and old joining data
    stitching_coord_dict = inout.load_stitching_coord(folder + data_name)
    # noinspection PyPep8Naming
    micData = stitching_coord_dict['micData']
    logger.info("Joining object and image list loaded from file")

    flat_tile_set = micData.tile_set.flat[:]
    image_list = [micData.tile_nr[ind] if ind >= 0 else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    logger.info("Tile set size: {}".format(micData.tile_set.shape))
    logger.info("Loading images: {}".format(image_list))

    # Make a list of the image names
    tiles = inout.get_image_names(tile_file, image_list = image_list,
                                hyb_nr = hyb_nr, gene = gene, pre_proc_level = pre_proc_level)

    contig_tuples   = stitching_coord_dict['contig_tuples']
    alignment_old   = stitching_coord_dict['alignment']['P']

    # Load the data from the metadata file
    ExperimentInfos, ImageProperties, HybridizationsInfos, \
    Converted_Positions, MicroscopeParameters = \
        utils.experimental_metadata_parser(folder)

    # Read the number of pixels and z-count from the yaml
    # file.
    try:
        nr_pixels = ImageProperties['HybImageSize']['rows']
    except KeyError as err:
        logger.info(("Number of pixels not found in experimental "
                     + "metadata file.\nPlease add "
                     + "the number of pixels in an image "
                     + "to the experimental "
                     + "metadata file under ImageProperties "
                     + "--> HybImageSize --> rows.\n"
                     + "KeyError: {}").format(err))
        raise

    if nr_dim == 2:
        z_count = 1
    else:
        try:
            z_count = ImageProperties['HybImageSize']['zcount']
        except KeyError as err:
            logger.info(
                ("Number of pixels not found in experimental "
                 + "metadata file.\nPlease add "
                 + "the number of slices in the z-stack "
                 + "to the experimental "
                 + "metadata file under ImageProperties "
                 + "--> HybImageSize --> zcount.\n"
                 + "KeyError: {}")
                    .format(err))
            raise


    # Recalculate C
    C = sklim.grid_to_graph(*micData.tile_set.shape).todense()
    np.fill_diagonal(C,0)
    C = np.triu( C )

    return (tiles, contig_tuples, nr_pixels, z_count, micData, C, alignment_old)


def refine_pairwise_alignments(tiles, tile_file, contig_tuples, alignment_old,
                                        micData = None, nr_peaks = 8,
                                        nr_dim = 2):
    """Calculate the pairwise transition

    Calculates pairwise transition for each neighbouring pair of
    tiles.

    Parameters:
    -----------

    tiles: np.array
        Array of tiles, a tile should be a 2d np.array
        representing a picture
    contig_tuples: list
        List of tuples denoting which tiles are contingent to each other.
    micData: object
        MicroscopeData object containing coordinates (default None)
    nr_peaks: int
        nr of peaks to be extracted from the PCM (default 8)
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (default: 2)

    Returns:
    --------
    : dict
        Contains key 'P' with a 1D numpy array
        containing pairwise alignment y and x coordinates
        (and z-coordinates when applicable) for each
        neighbouring pair of tiles, array will be
        2 * len(contig_typles) for 2D data
        or 3 * len(contig_typles) for 3D data.
        Also contains key 'covs' with a 1D numpy array
        containing covariance for each pairwise alignment in
        'P', 'covs' will be len(contig_typles).
    """
    # Make a new P and cov list for the refine pairwise alignments
    P_ref = np.empty((len(contig_tuples), nr_dim), dtype = int)
    covs_ref = np.empty(len(contig_tuples))

    for i in range(len(contig_tuples)):
        P_single, cov, contig_index  = ps.refine_single_pair(tiles,
                                        tile_file,
                                        contig_tuples, i, micData,
                                        alignment_old['P'], nr_peaks,
                                        nr_dim = nr_dim)
        P_ref[contig_index,:] = P_single
        covs_ref[contig_index] = cov

    logger.info("Raw P: {}".format(P_ref))
    # Flatten P
    P_ref = np.array(P_ref).flat[:]
    logger.info("flat P: {}".format(P_ref))
    return {'P': P_ref, 'covs': covs_ref}


######################### General functions ############################
def get_place_tile_input(folder, tiles, contig_tuples,
                            micData, nr_pixels, z_count, alignment,
                            data_name, nr_dim = 2, save_alignment = True):
    """Do the global alignment and get the shifted corner coordinates.

    Calculates a shift in global coordinates for each tile (global
    alignment) and then applies these shifts to the  corner coordinates
    of each tile and returns and saves these shifted corner coordinates.

    This function produces a file with stitching data in folder
    called data_name, this file includes the corner coordinates which
    can be used to apply the stitching to another gene.

    Parameters:
    -----------

    folder: str
        String representing the path of the folder containing
        the tile file and the yaml metadata file. Needs a
        trailing slash ('/').
    tiles: list
        List of strings. List of references to the the
        tiles in the hdf5 file tile_file.
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    micData: object
        MicroscopeData object. Contains coordinates of
        the tile corners as taken from the microscope.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of the z-axis). Is 1 when nr_dim is not 3.
    alignment: dict
        Contains key 'P' with a 1D numpy array
        containing pairwise alignment y and x coordinates
        (and z-coordinates when applicable) for each
        neighbouring pair of tiles, array will be
        2 * len(contig_typles) for 2D data
        or 3 * len(contig_typles) for 3D data.
        Also contains key 'covs' with a 1D numpy array
        containing covariance for each pairwise alignment in
        'P', 'covs' will be len(contig_typles).
    data_name: str
        Name of the file containing the pickled stitching data.
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (default: 2)
    save_alignment: bool
        When False only the stitching
        coordinates and microscope data will be saved. When
        True also the contigency tuples and pairwise
        alignment will be saved (this is necessary if we
        want to refine the stitching later). (Default: True)

    Returns:
    --------
    joining: dict
        Contains keys corner_list and
        final_image_shape.
        Corner_list is a list of list, each list is a pair
        of a tile index (int) and it's tile's shifted
        coordinates in the final image (numpy array
        containing floats).
        Final_image_shape is a tuple of size 2 or 3
        depending on the numer of dimensions and contains
        ints.

    """
    # Perform global optimization
    logger.debug("Initializing global optimization")
    optimization = GlobalOptimization()
    logger.debug("Starting optimization, micData")
    optimization.performOptimization(micData.tile_set, contig_tuples,
                                alignment['P'], alignment['covs'],
                                len(tiles), nr_dim)


    # Stitch everything back together
    # Determine global corners
    joining = tilejoining.calc_corners_coord(tiles,
                optimization.global_trans, micData, nr_pixels, z_count)

    # Save the data to do the stitching in "data_name":
    if joining:
        if save_alignment:
            inout.save_to_file(folder + data_name,
                               joining = joining,
                               contig_tuples = contig_tuples,
                               alignment = alignment,
                               micData = micData)
        else:
            inout.save_to_file(folder + data_name,
                                micData = micData,
                                joining = joining)
    else:
        logger.warning("No results found to save: joining is empty")

    return joining

def assess_performance(micData, alignment, joining,
                        cov_signal, xcov_list, folder,
                        use_IJ_corners = False):
    """Assess the performance of the stitching

    This functions writes its the result to a file in "folder".

    Parameters:
    -----------

    micData: object
        MicroscopeData object. Contains coordinates of
        the tile corners as taken from the microscope
        and contains the tile set.
    alignment: dict
        Contains key 'P' with a 1D numpy array
        containing pairwise alignment y and x coordinates
        (and z-coordinates when applicable) for each
        neighbouring pair of tiles, array will be
        2 * len(contig_typles) for 2D data
        or 3 * len(contig_typles) for 3D data.
        Also contains key 'covs' with a 1D numpy array
        containing covariance for each pairwise alignment in
        'P', 'covs' will be len(contig_typles).
    joining: dict
        Contains keys corner_list and
        final_image_shape.
        Corner_list is a list of list, each list is a pair
        of a tile index (int) and it's tile's shifted
        coordinates in the final image (numpy array
        containing floats).
        Final_image_shape is a tuple of size 2 or 3
        depending on the numer of dimensions and contains
        ints.
    cov_signal: np.array
        The covariance of each neighbouring
        tile pair in the part of the tiles that overlap in
        the final stitched signal image.
    xcov_list: list
        List of cross covariance of the
        overlap of the tiles in the final stitched image.
        As returned by tilejoining.assess_overlap.
    folder: str
        String representing a path. The folder where the
        performance report should be saved. Needs a
        trailing slash ('/').
    use_IJ_corners: bool
        If True compare our corners to Image J
        found in a file in folder, the file name should
        contain: TileConfiguration.
    """
    ################################Gather data#########################
    report_string   = ""
    if use_IJ_corners:
        compare_corners = inout.read_IJ_corners(folder)
        # Make a list of image numbers, matching with the numbers in the
        # image files
        flat_tile_set = micData.tile_set.flat[:]
        image_list = [micData.tile_nr[ind] if ind >= 0
                      else -1 for ind in flat_tile_set]
        image_list = np.ma.masked_equal(image_list, -1)
        # Compare our corners to the corner the ImageJ plugin found
        # Select the tiles we actually used:
        compare_corners_new = [[item[0], np.array([item[1][1], item[1][0]])]
                                for item in compare_corners
                                if item[0] in image_list]
        # Replace the indexes with numbering of the images:
        logger.debug('My corners {}'.format(joining['corner_list']))
        my_corners_new = [[image_list[i], item[1]] for i, item in enumerate(joining['corner_list'])]
        logger.debug('My corners new {}'.format(my_corners_new))
        logger.debug('Compare corners new {}'.format(compare_corners_new))
        # Normalize, to make the first one the origin and compare
        compare_origin  = compare_corners_new[0][1]
        my_origin       = my_corners_new[0][1]
        logger.debug("origins: {}, {}".format(compare_origin, my_origin))

        cum_diff = np.zeros((1,2))
        for i in range(len(my_corners_new)):
            my_cur = (my_corners_new[i][1] - my_origin)
            compare_cur = (compare_corners_new[i][1] - compare_origin)
            diff = abs(my_cur - compare_cur)
            cum_diff += diff
            report_string += ("My tile: {}, {}, compare tile: {}, {}; difference: {}\n"
                                .format(my_corners_new[i][0], my_cur,
                                        compare_corners_new[i][0], compare_cur,
                                        diff))
        report_string += "\nAverage: {}\n".format(cum_diff / len(my_corners_new))

    # Calculate average cross covariances of the overlaps in the final image:
    if xcov_list is not None:
        av_xcov = np.mean(xcov_list)
    else:
        logger.info('No cross covariance data available')
        av_xcov = None
    report_string += "\nAverage cross covariance of final overlap: {}\n".format(av_xcov)
    report_string += "Cross covariance list of final overlap: {}\n".format(xcov_list)
    #logger.debug(report_string)

    ###################### Save the performance data ###################
    perf_path = folder + 'performance/'
    # To print the logging to a file
    try:
            os.stat(perf_path)
    except:
            os.mkdir(perf_path)
            os.chmod(perf_path,0o777)

    dateTag = time.strftime("%y%m%d_%H_%M_%S")
    with open(perf_path + dateTag + '-performance' + '.txt', 'w') as f:
        f.write(report_string)
        if micData is not None:
            # If available print the alignment results:
            f.write(("\nTile set: \n{} \n"
                    "Tile numbers: {}\n")
                    .format(micData.tile_set, micData.tile_nr))
        if alignment is not None:
            # If available print the alignment results:
            f.write(("Pairwise Alignment: {}\n"
                    "Covariances: {}\n"
                    "Average covariance: {}\n")
                    .format(alignment['P'], alignment['covs'],
                            np.nanmean(alignment['covs'])))
        if joining is not None:
            f.write(("Corners after alignment: \n{}\n")
                    .format(joining['corner_list']))
        if cov_signal is not None:
            f.write(("\nAverage pairwise covariance of the signal: {}\n"
                     "Pairwise covariance of the signal:\n{}\n")
                                            .format(np.nanmean(cov_signal), cov_signal))
    f.close()


############################# Visualization ############################
def save_as_tiff(data_file, hyb_nr, gene, location_image,
                    pre_proc_level = 'StitchedImage', mode = 'both'):
    """Save the results as a tiff image for visual inspection.

    Parameters:
    -----------

    data_file: pointer
        HDF5 file handle. HDF5 file containing the final image.
    gene: str
        The name of the gene we stitched.
        This will be used to navigate data_file and find the
        correct final picture.
    hyb_nr: int
        The number of the hybridization we have
        stitched.This will be used to navigate data_file and
        find the correct final picture.
    location_image: str
        Full path to the file where the tiff file
        will be saved (extension not necessary).
    pre_proc_level: str
        The name of the pre processing group of
        the tiles we are going to stitch. Normally this will
        be 'StitchedImage', but when the final image is
        found in another datagroup it may be changed.
        This will be used to navigate data_file and find the
        correct final image. (Default: 'StitchedImage')
    mode: str
        Mode determines what color, quality and
        how many images are saved.
        Possible values for mode: save_ubyte, save_float,
        save_rgb. If another or no value is given the image
        is saved as is and a as a low quality copy
        (pixel depth 8 bits) (Default: 'both')
    """
    # Save the results:
    if mode == 'save_ubyte':
        inout.save_image(data_file, hyb_nr, gene, pre_proc_level,  'final_image_ubyte', location_image + '_byte')
    elif mode == 'save_float':
        inout.save_image(data_file, hyb_nr, gene, pre_proc_level, 'final_image', location_image)
    elif mode == 'save_rgb':
        inout.save_image(data_file, hyb_nr, gene, pre_proc_level, 'final_image_rgb', location_image + '_rgb')
    else:
        inout.save_image(data_file, hyb_nr, gene, pre_proc_level, 'final_image_ubyte', location_image + '_byte')
        inout.save_image(data_file, hyb_nr, gene, pre_proc_level, 'final_image', location_image)


def plot_final_image(im_file_name, joining,  hyb_nr = 1,
                     gene = 'Nuclei', fig_name = "final image",
                     shrink_image = False, block = True):
    """Displays the high quality final image in a plot window.

    Takes a lot of working memory for full sized images.
    When plt_available is false this function does nothing and returns
    None.

    Parameters:
    -----------

    im_file_name: str
        Filename of the hdf5 file, containing the final image.
    fig_name: str
        Name of the plotting window (default: "final image").
    shrink_image: bool
        Turn on shrink_image to reduce display quality and memory usage. (Default: False)
    block: bool
        Plot blocks the running program untill
        the plotting window is closed if true. Turn off
        block to make the code continue untill the next call
        of  plt.show(block=True) before displaying the
        image. (default: True)
    """
    if plt_available:
        if isinstance(im_file_name, str):
            # Load the image from file
            im_file = h5py.File(im_file_name + '_Hybridization' +
                                str(hyb_nr) + '.sf.hdf5', 'r')
            for_display = im_file['final_image']
        else:
            # Load the image from file
            for_display = im_file_name[gene] \
                                    ['StitchedImage']['final_image']
        # Shrink the image if necessary
        if shrink_image:

            display_size = np.array(joining['final_image_shape'],
                                            dtype = int)/10
            logger.debug("display size pixels: {}".format(display_size))
            for_display = smtf.resize(for_display, tuple(display_size))
        # Plot the image
        if for_display.ndim == 3:
            inout.plot_3D(for_display)
        else:
            plt.figure(fig_name)
            plt.imshow(for_display, 'gray', interpolation = 'none')
            plt.show(block = False)

        # Load the image from file
        if isinstance(im_file_name, str):
            # Load the image from file
            im_file = h5py.File(im_file_name + '.hdf5', 'r')
            for_display = im_file['temp_mask']
        else:
            for_display = im_file_name['Hybridization' + str(hyb_nr)][gene] \
                        ['StitchedImage']['temp_mask']
        # Shrink the image if necessary
        if shrink_image:
            display_size = np.array(joining.final_image_shape, dtype=int) / 10
            logger.debug("display size pixels: {}".format(display_size))
            for_display = smtf.resize(for_display, tuple(display_size))
        # Plot the image
        plt.figure(fig_name + ' mask')
        plt.imshow(for_display, 'gray', interpolation='none')
        plt.show(block = block)
    else:
        return None


def get_pairwise_input_npy(image_properties,converted_positions, hybridization,
                        est_overlap = 0.1, y_flip = False, nr_dim = 2):
    """Get the information necessary to do the pairwise allignment
    Modified version of the get_pairwise_input functions that work on .npy 
    files and not on hdf5

    Find the pairwise pairs for an unknown stitching.
    
    Parameters:
    -----------
    
    image_properties: dict 
        Dictionary with the image details parsed from the Experimental_metadata.yaml file
    converted_positions: dict
        Dictionary  with the coords of the images for all hybridization
        The coords are a list of floats
    hybridization: str 
        Hybridization that will be processed (Ex. Hybridization2)
    est_overlap: float
        The fraction of two neighbours that should
        overlap, this is used to estimate the shape of the
        tile set and then overwritten by the actual average
        overlap according to the microscope coordinates.
        (default: 0.1)
    y_flip: bool
        The y_flip variable is designed for the cases where the
        microscope sequence is inverted in the y-direction. When
        set to True the y-coordinates will also be inverted
        before determining the tile set. (Default: False)
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (Default: 2)
    
    Returns:
    --------

    tiles: np.array
        Array of int with the tiles number. -1 indicate an empty tile
    contig_tuples: list
        List of tuples. Each tuple is a tile pair.
        Tuples contain two tile indexes denoting these
        tiles are contingent to each other.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of
        the z-axis). Is 1 when nr_dim is not 3.
    micData: object
        MicroscopeData object. Contains coordinates of
        the tile corners as taken from the microscope.
    """


    # Get coordinate data for this hybridization
    coord_data = converted_positions[hybridization]

    # Read the number of pixels, z-count and pixel size from the yaml
    # file.
    try:
        nr_pixels       = image_properties['HybImageSize']['rows']
    except KeyError as err:
        logger.info(("Number of pixels not found in experimental "
                     + "metadata file.\nPlease add "
                     + "the number of pixels in an image "
                     + "to the experimental "
                     + "metadata file under ImageProperties "
                     + "--> HybImageSize --> rows.\n"
                     + "KeyError: {}").format(err))
        raise

    if nr_dim == 2:
        z_count = 1
    else:
        try:
            z_count = image_properties['HybImageSize']['zcount']
        except KeyError as err:
            logger.info(("Number of pixels not found in experimental "
                         + "metadata file.\nPlease add "
                         + "the number of slices in the z-stack "
                         + "to the experimental "
                         + "metadata file under ImageProperties "
                         + "--> HybImageSize --> zcount.\n"
                         + "KeyError: {}")
                         .format(err))
            raise

    try:
        pixel_size = image_properties['PixelSize']
    except KeyError as err:
        logger.info(("ImageProperties['PixelSize'] not found in "
                    + "experimental metadata file.\nPlease add the "
                    + "size of a pixel in um in the experimental "
                    + "metadata file under ImageProperties "
                    + "--> PixelSize.\nKeyError: {}").format(err))
        raise
    # Estimate the overlap in pixels with the overlap that the user
    # provided, default is 10%
    est_x_tol = nr_pixels * (1 - est_overlap)
    logger.info("Estimating overlap at {}%, that is {} pixels"
                .format(est_overlap * 100, est_x_tol))
    logger.debug("Number of pixels: {}".format(nr_pixels))
    logger.debug("Number of slices in z-stack: {}".format(z_count))

    # Organize the microscope data and determine tile set
    micData = MicroscopeData(coord_data, y_flip, nr_dim)
    micData.normalize_coords(pixel_size)
    micData.make_tile_set(est_x_tol, nr_pixels = nr_pixels)

    # Make a list of image numbers, matching with the numbers in the
    # image files
    flat_tile_set = micData.tile_set.flat[:]

    image_list = [micData.tile_nr[ind] if ind >= 0 else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    logger.info("Getting references for: {}".format(image_list))


    # Make a list of the image names (-1 is a missing tile)
    tiles = image_list.data


    # Produce an undirected graph of the tiles, tiles that are
    # neighbours to each other are connected in this graph.
    # noinspection PyPep8Naming
    C = np.asarray(sklim.grid_to_graph(*micData.tile_set.shape).todense())
    np.fill_diagonal(C, 0)
    # noinspection PyPep8Naming
    C = np.triu( C )
    # Extract the neighbour pairs from the graph
    contig_tuples =list(zip( *np.where( C ) ))
    logger.info(("Length contingency tuples: {} \n"
                + "Contingency tuples: {}")
                 .format(len(contig_tuples), contig_tuples))

    return(tiles, contig_tuples, nr_pixels, z_count, micData)


def get_place_tile_input_apply_npy(hyb_dir,stitched_reference_files_dir,data_name,image_properties,nr_dim=2):
    """
    Modified version of the get_place_tile_input_apply
    Get the data needed to apply stitching to another gene

    Parameters:
    -----------

    hyb_dir: str
        String representing the path of the folder containing
        the tile file, the stitching data file the yaml metadata file.
    stitched_reference_files_dir: str
        String representing the path of the folder containing the registered data.
    data_name: str  
        Name of the file containing the pickled stitching data.
    image_properties: dict
        Dictionary with the image details parsed from the Experimental_metadata.yaml file
    nr_dim: int
        If 3, the code will assume three dimensional data
        for the tile, where z is the first dimension and y and x
        the second and third. For any other value 2-dimensional data
        is assumed. (Default: 2)
    
    Returns:
    --------

    joining: dict
        Taken from the stitching data file.
        Contains keys corner_list and final_image_shape.
        Corner_list is a list of list, each list is a pair
        of an image number (int) and it's coordinates (numpy
        array containing floats).
        Final_image_shape is a tuple of size 2 or 3
        depending on the numer of dimensions and contains
        ints.
    tiles: list
        List of strings. List of references to the the tiles in the hdf5 file tile_file.
    nr_pixels: int
        Height and length of the tile in pixels, tile is assumed to be square.
    z_count: int
        The number of layers in one tile (size of the z-axis). Is 1 when nr_dim is not 3.
    micData: object
        MicroscopeData object. Taken from the pickled stitching data.
        Contains coordinates of the tile corners as taken from the microscope.
    """
    
    logger.info("Getting data to apply stitching from file...")

    # Load image list and old joining data
    stitching_coord_dict = inout.load_stitching_coord(stitched_reference_files_dir + data_name)
    # noinspection PyPep8Naming
    micData = stitching_coord_dict['micData']
    joining = stitching_coord_dict['joining']
    logger.info("Joining object and image list loaded from file")
    
    # Make a list of image numbers
    flat_tile_set = micData.tile_set.flat[:]
    image_list = [micData.tile_nr[ind] if ind >= 0 else -1 for ind in flat_tile_set]
    image_list = np.ma.masked_equal(image_list, -1)
    logger.info("Tile set size: {}".format(micData.tile_set.shape))
    logger.info("Placing folowing image references in tiles: {}"
                .format(image_list))
    
    # Make a list of the image names (-1 is a missing tile)
    tiles = image_list.data
    
    
    # Read the number of pixels and z-count from the yaml
    # file.
    try:
        nr_pixels = image_properties['HybImageSize']['rows']
    except KeyError as err:
        logger.info(("Number of pixels not found in experimental "
                     + "metadata file.\nPlease add "
                     + "the number of pixels in an image "
                     + "to the experimental "
                     + "metadata file under ImageProperties "
                     + "--> HybImageSize --> rows.\n"
                     + "KeyError: {}").format(err))
        raise

    if nr_dim == 2:
        z_count = 1
    else:
        try:
            z_count = image_properties['HybImageSize']['zcount']
        except KeyError as err:
            logger.info(
                ("Number of pixels not found in experimental "
                 + "metadata file.\nPlease add "
                 + "the number of slices in the z-stack "
                 + "to the experimental "
                 + "metadata file under ImageProperties "
                 + "--> HybImageSize --> zcount.\n"
                 + "KeyError: {}")
                .format(err))
            raise

    logger.info("Size tiles: {} Number of pixels: {} z count: {}"
                .format(len( tiles), nr_pixels, z_count))


    return (joining, tiles, nr_pixels, z_count, micData)
