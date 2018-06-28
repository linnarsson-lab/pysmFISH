"""Functions to perform registration between all hybridizations.

register_final_images(folder, gene='Nuclei',
                      sub_pic_frac=0.2, use_MPI=False,
                      apply_to_corners=True, apply_warping = True)
        -- Register stitched images an in all  HDF5 file in the folder
find_reg_final_image(im_file_1, im_file_n,
                           max_trans, sub_pic_frac,
                           nr_peaks=8)
        -- Find the transform that registers image n correctly onto
        image 1.
transform_final_image(im_file_n, trans, new_size)
        -- Transform an image according to trans.
transform_data_file(folder, data_name, trans,
                        new_size)
        -- Transform the corners in the pickled data file
align_sub_region(overlap1, overlap2, nr_peaks)
        -- Determine how much overlap2 should be shifted to fit
        overlap1, help function for find_reg_final_image
"""

import numpy as np
import h5py
import os
import skimage.transform as smtf
try:
    from mpi4py import MPI
    MPI_available = True
except ImportError:
    MPI_available = False
import logging
import glob

# Own imports
from . import inout
from . import pairwisesingle as ps

logger = logging.getLogger(__name__)


def register_final_images(folder, gene='Nuclei',
                          sub_pic_frac=0.2, use_MPI=False,
                          apply_to_corners=True, apply_warping = False,
                          region=None, compare_in_seq=False):
    """Register stitched images an in all  HDF5 file in the folder

    Loops the hybridizations in the HDF5 file, takes the stitched
    images as indicated by gene and then compares each image to the
    first image.
    For the comparison only a small patch of the images is used, the
    size of this patch can be controlled with "sub_pic_frac".

    Parameters:
    -----------

    folder: str
        The name of the folder containing the pickled file with stitching data, 
        needs a trailing slash ("/").
    gene: str 
        The gene of which the stitched images are present and should be realigned.
        Typically this will be 'Nuclei', because the smFISH genes will not have 
        enough signal to align the pictures properly. (Default: 'Nuclei')
    sub_pic_frac: float 
        The fraction of the size of the original image that should be used to compare
        images. (Default: 0.2)
    use_MPI: bool 
        If True open the files in MPI friendly mode, if False open files in normal
        single processing mode. (Default: False)
    apply_to_corners: bool
        Determines if the found registration will be applied to the tile
        corners in the pickled stitching data file. (Default: True)
    apply_warping: bool
        Determines if the found registration will be applied as a warp to the
        final pictures in the hdf5 file, should not be used with large datasets.
        (Default: False)
    region: list 
        List of length four containing ints. The region that should be compared to determine
        the shift needed for registration. Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac will not be used.
        By default the code will determine the region itself taking a area around the
        center of the image with a size determined by sub_pic_frac(Default: None)
    compare_in_seq: bool 
        Determines if we should compare images in sequence or if we should compare
        all to the first image.

    """
    if not compare_in_seq:
        file_name_list, file_1, im_file_1, trans, old_size_list, \
            max_trans = \
            prepare_for_comparing(folder, gene, compare_in_seq,
                                  use_MPI=use_MPI)
        # Compare each file to file 1:
        for i in range(1, len(file_name_list)):
            cur_trans, max_trans, cur_old_size, file_ind = \
                get_single_trans(file_name_list, i, gene, im_file_1,
                                 max_trans, sub_pic_frac=sub_pic_frac,
                                 region=region, use_MPI=use_MPI)
            trans[file_ind, :] = cur_trans
            old_size_list[file_ind, :] = cur_old_size
        # Close the hdf5 file.
        file_1.close()
        trans, new_size = correct_trans_and_size(trans,
                                                 old_size_list,
                                                 max_trans,
                                                 compare_in_seq)
    else:
        file_name_list, trans_relative, old_size_list, max_trans = \
            prepare_for_comparing(folder, gene, compare_in_seq,
                                  use_MPI=use_MPI)
        # Compare each file to previous file:
        for i in range(1, len(file_name_list)):
            cur_trans, max_trans, cur_old_size, file_ind = \
                get_single_relative_trans(file_name_list, i, gene,
                                          max_trans,
                                          sub_pic_frac = sub_pic_frac,
                                          region = region,
                                          use_MPI = use_MPI)
            trans_relative[file_ind, :] = cur_trans
            old_size_list[file_ind, :]  = cur_old_size
        trans, new_size = correct_trans_and_size(trans_relative,
                                                 old_size_list,
                                                 max_trans,
                                                 compare_in_seq)

    logger.debug(
        'Files: {} Translations: {}'
            .format(file_name_list, trans))

    # Apply the translations
    for i in range(len(file_name_list)):
        if apply_warping:
            if use_MPI:
                file_n = h5py.File(file_name_list[i], 'r+',
                                   driver='mpio', comm=MPI.COMM_WORLD)
            else:
                file_n = h5py.File(file_name_list[i], 'r+')
            im_file_n = file_n[gene]['StitchedImage']
            transform_final_image(im_file_n, trans[i, :], new_size)
            file_n.close()
        if apply_to_corners:
            data_name = (
            os.path.split(file_name_list[i])[1].split(sep='.')[0]
            + '_' + gene
            + '_stitching_data')
            transform_data_file(folder, data_name, trans[i, :],
                                new_size)

def prepare_for_comparing(folder, gene, compare_in_seq, use_MPI=False):
    """
    Prepare the file list, first file and init other lists.

    Parameters:
    -----------

    folder: str         
        The name of the folder containing the
        pickled file with stitching data, needs a
        trailing slash ("/").
    gene: str
        The gene of which the stitched
        images are present and should be realigned.
        Typically this will be 'Nuclei', because the
        smFISH genes will not have enough signal to
        align the pictures properly.
        (Default: 'Nuclei')
    compare_in_seq: bool
        Determines if we should compare
        images in sequence or if we should compare
        all to the first image.
    use_MPI: bool
        If True open the files in MPI
        friendly mode, if False open files in normal
        single processing mode. (Default: False)
    
    Returns:
    --------
    file_name_list: list
        List of strings. List of the sf.hdf5-files in the folder.
    trans: np.array
        Array of ints. The array to store the translations, initialized with zeros.
    old_size_list: np.array 
        Array of ints. The array to store the sizes of the final images, initialized with
        zeros.
    max_trans: np.array
        Array of ints. Variable to store the
        largest translation found up to now,
        initialized at zero.
    
    Notes:
    ------
    Only returned if compare_in_seq is True:
    file_1: pointer
        File handle to the first hdf5 file in the folder.
    im_file_1: pointer
        Reference to the group in   the first file that contains th final image.
    """
    # Get a list of files in the folder
    file_name_list = glob.glob(folder + '*.sf.hdf5')
    file_name_list.sort()
    logger.debug('Filenames sorted: {}'.format(file_name_list))
    # Initialize some variables:
    trans = np.zeros((len(file_name_list), 2), dtype=int)
    old_size_list = np.zeros((len(file_name_list), 2), dtype=int)
    max_trans = np.zeros((1, 2), dtype=int)

    # Take the first hybridization (keys() seems to give the groups
    # as a sorted list)
    im_name_1 = file_name_list[0]
    logger.debug('im_name_1: {}'.format(im_name_1))
    # Open the stitching file and make a list of the hybridizations
    # present in this file:
    if use_MPI:
        file_1 = h5py.File(im_name_1, 'r+',
                           driver='mpio', comm=MPI.COMM_WORLD)
    else:
        file_1 = h5py.File(im_name_1, 'r+')

    # hyb_name_list = list(stitching_file.keys())

    # Get the right group
    im_file_1 = file_1[gene]['StitchedImage']
    # Get the size of the first image in the list,
    # which will be the reference image without translation.
    old_size_list[0, :] = im_file_1['final_image'].shape
    # Make comparisons
    if not compare_in_seq:
        return file_name_list, file_1, im_file_1, trans, \
               old_size_list, max_trans
    else:
        # Get the size for the first image
        file_1.close()
        return file_name_list, trans, old_size_list, max_trans


def get_single_trans(file_name_list, i, gene,
                     im_file_1, max_trans, sub_pic_frac=0.2,
                     region=None, use_MPI=False):
    """Get the translation between image 1 and image i.

    Get the translation between the image in file 1 and file i
    from file_name_list.

    Parameters:
    -----------

    file_name_list: list
        List of strings. List of the sf.hdf5-files in the folder.
    i: int
        Index of the current file to compare.
    gene: str
        Gene of which the stitched
        images are present and should be realigned.
        Typically this will be 'Nuclei', because the
        smFISH genes will not have enough signal to
        align the pictures properly.
        (Default: 'Nuclei')
    im_file_1: pointer
        Reference to the group in the first file that contains th final image.
    max_trans: np.array
        Variable to store the largest translation found up to now,
        initialized at zero.
    sub_pic_frac: float
        The fraction of the size of the original image that should be used to compare
        images. (Default: 0.2)
    region: list
        List of length four containing ints. The
        region that should be compared to determine
        the shift needed for registration.
        Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac
        will not be used.
        By default the code will determine the region
        itself taking a area around the
        center of the image with a size
        determined by sub_pic_frac(Default: None)
    use_MPI: bool
        If True open the files in MPI friendly mode, if False open files in normal
        single processing mode. (Default: False)

    Returns:
    --------

    cur_trans: np.array
        Array of ints. Translation found between the two images that are currently
        being compared.
    max_trans: np.array 
        Array of ints. The largest translation found up to now.
    cur_old_size: np.array
        Array of ints. The sizes of the original final image found in file_name_list
        at index i.
    i: int
        The index of the second image file used for the current comparison (The first image
        file is file 1).
    """
    # Get the group containing the image we want to compare with.
    if use_MPI:
        file_n = h5py.File(file_name_list[i], 'r+',
                           driver='mpio', comm=MPI.COMM_WORLD)
    else:
        file_n = h5py.File(file_name_list[i], 'r+')
    im_file_n = file_n[gene]['StitchedImage']
    # Find the translation
    cur_trans, max_trans, cur_old_size \
        = find_reg_final_image(im_file_1, im_file_n, max_trans,
                               sub_pic_frac, region=region)
    logger.debug(
        "max_trans: {}".format(max_trans))
    file_n.close()
    return cur_trans, max_trans, cur_old_size, i

def get_single_relative_trans(file_name_list, i, gene, max_trans,
                              sub_pic_frac=0.2, region=None,
                              use_MPI=False):
    """Get the translation between image i - 1 and image i.

    Get the translation between the image in file_name_list[i - 1] and
    file_name_list[i].

    Parameters:
    -----------

    file_name_list: list
        List of strings. List of the sf.hdf5-files in the folder.
    i: int
        Index of the second image in the current comparison.
    gene: str
        The gene of which the stitched
        images are present and should be realigned.
        Typically this will be 'Nuclei', because the
        smFISH genes will not have enough signal to
        align the pictures properly.
        (Default: 'Nuclei')
    max_trans: np.array
        Array of ints. Variable to store the
        largest translation found up to now,
        initialized at zero.
    sub_pic_frac: float
        The fraction of the size of the original image that should be used to compare
        images. (Default: 0.2)
    region: list
        List of length four containing ints. The
        region that should be compared to determine
        the shift needed for registration.
        Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac
        will not be used.
        By default the code will determine the region
        itself taking a area around the
        center of the image with a size
        determined by sub_pic_frac (Default: None)
    use_MPI: bool
        If True open the files in MPI
        friendly mode, if False open files in normal
        single processing mode. (Default: False)

    Returns:
    --------

    cur_trans: np.array
        Array of ints. Translation found between the two images that are currently
        being compared.
    max_trans: np.array
        Array of ints. The largest translation found up to now.
    cur_old_size: np.array
        The sizes of the original final image found in file_name_list
        at index i.
    i: int
        The index of the second image file used for the current comparison (The first image
        file is file 1).

    """
    # Get the group containing the image we want to compare with.
    if use_MPI:
        file_1 = h5py.File(file_name_list[i - 1], 'r+',
                           driver='mpio', comm=MPI.COMM_WORLD)
        file_2 = h5py.File(file_name_list[i], 'r+',
                           driver='mpio', comm=MPI.COMM_WORLD)
    else:
        file_1 = h5py.File(file_name_list[i - 1], 'r+')
        file_2 = h5py.File(file_name_list[i], 'r+')
    im_file_1 = file_1[gene]['StitchedImage']
    im_file_2 = file_2[gene]['StitchedImage']
    # Find the translation
    cur_trans, max_trans, cur_old_size \
        = find_reg_final_image(im_file_1, im_file_2, max_trans,
                               sub_pic_frac, region=region)
    logger.debug("max_trans: {}".format(max_trans))
    file_1.close()
    file_2.close()
    return cur_trans, max_trans, cur_old_size, i


def correct_trans_and_size(trans_relative, old_size_list, max_trans,
                           compare_in_seq):
    """Correct the translations and the size of the registered images.

    Parameters:
    -----------

    trans_relative: np.array
        Array of ints. The array with the non-corrected translation.
    old_size_list: np.array
        Array of ints. The array with the sizes of all the final,
        non-registered images.
    max_trans: np.array
        Array of ints. Variable to store the largest translation found up to now.
    compare_in_seq: bool
        Determines if we should compare images in sequence or if we should compare
        all to the first image.

    Returns:
    --------

    trans: np.array
        Array of ints. The array with the corrected translations for each image.
    new_size: np.array 
        Array of length 2 containing ints. The size the images should have after registration.
    """
    if compare_in_seq:
        # Get the normalized transistions
        trans = np.cumsum(trans_relative, axis=0)
        max_trans = np.amax(trans, axis=0)
        logger.debug(("Comparing in sequence: relative translations: "
                      + "\n {} \n normalized translations: \n{}\n"
                      .format(trans_relative, trans)))
        logger.debug("max_trans: {}".format(max_trans))
    else:
        trans = trans_relative

    # Correct translations
    trans -= max_trans
    logger.debug('old_size_list: {}'
                 .format(old_size_list))
    # Determine final image size:
    new_size_list = old_size_list + abs(trans)
    new_size = np.amax(new_size_list, axis=0)
    logger.debug('new_size_list: {}  new_size: {}'
                 .format(new_size_list, new_size))
    return trans, new_size


def find_reg_final_image(im_file_1, im_file_n, max_trans, sub_pic_frac,
                         region=None, nr_peaks=8):
    """
    Find the transform that registers image n correctly onto image 1.
    
    Parameters:
    im_file_1: pointer
        HDF5 group reference or file handle, should
        contain a dataset "final_image" holding image 1.
    im_name_n: pointer
        HDF5 group reference or file handle, should
        contain a dataset "final_image" holding image n.
    max_trans: np.array
        Array of length 2 with dtype: int.
        Largest translation currently found.
    sub_pic_frac: float 
        The fraction of the size of the original 
        image that should be used to compare images.
    region: list
        List of length four containing ints. The
        region that should be compared to determine
        the shift needed for registration.
        Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac
        will not be used.
        By default the code will determine the region
        itself taking a area around the
        center of the image with a size
        determined by sub_pic_frac(Default: None)
    nr_peaks: int        
        The number of peaks used to get the best peaks 
        from the phase correlation matrix. (default: 8)

    Returns:
    --------

    trans: np.array
        Array of length 2 containing ints.
        Translation that projects image n correctly onto image 1.
    max_trans: np.array
        Array of shape (1, 2) containing ints.
        The max_trans value that was passed to this
        function, replaced by (part of) the current
        translation if it is larger than max_trans.
    shape_n: tuple
        Tuple of python ints. The shape of image n.
    """
    # Get the image shapes
    shape_1 = im_file_1['final_image'].shape
    shape_n = im_file_n['final_image'].shape

    if region is None:
        # Determine the size of the part of the picture that we want to compare
        sub_pic_size = (np.array(shape_1) * sub_pic_frac).astype(int,
                                                                 copy=False)
        logger.debug('sub_pic_size: {}'.format(sub_pic_size))
        # Take the center coordinates in the y and x axes
        center = (int(np.floor(min(shape_1[-2] / 2,
                                   shape_n[-2] / 2))),
                  int(np.floor(min(shape_1[-1] / 2,
                                   shape_n[-1] / 2))))
        start = np.array([center[0], center[1]])
        end = np.array([min(start[-2] + sub_pic_size[-2],
                            shape_1[-2],
                            shape_n[-2]),
                        min(start[-1] + sub_pic_size[-1],
                            shape_1[-1],
                            shape_n[-1])])
    else:
        start   = np.array([region[0],region[2]])
        end     = np.array([region[1],region[3]])
        logger.debug("Area based on given region: Start: {} "
                     "End: {}".format(start, end))
    # Get the region to compare from the pictures
    if im_file_1['final_image'].ndim == 3:
        pic_1 = np.amax(im_file_1['final_image'][:, start[0]:end[0],
                        start[1]:end[1]])
    else:
        pic_1 = im_file_1['final_image'][start[0]:end[0],
                start[1]:end[1]]
    if im_file_1['final_image'].ndim == 3:
        pic_n = np.amax(im_file_n['final_image'][:, start[0]:end[0],
                        start[1]:end[1]])
    else:
        pic_n = im_file_n['final_image'][start[0]:end[0],
                start[1]:end[1]]
    # Find the best translation
    trans, best_cov = align_sub_region(pic_1, pic_n, nr_peaks)
    logger.debug('Found trans: {} \n best covariance: {}'
                 .format(trans, best_cov))
    # Adjust max trans if necessary
    max_trans = np.maximum(max_trans, np.array(trans))

    return trans, max_trans, shape_n


def transform_final_image(im_file_n, trans, new_size):
    """
    Transform an image according to trans.
    
    Parameters:
    -----------

    im_file_n: pointer 
        HDF5 group reference or file handle, should
        contain a dataset "final_image" holding image n.
    trans: np.array
        Array of len 2 containing ints. y and x transform of the image.
    new_size: tuple
        Tuple of length 2. The size of the image after the transform.
    """
    # Make the trans matrix
    trans_matrix = np.eye(3)
    trans_matrix[1][2] = trans[0]
    trans_matrix[0][2] = trans[1]

    # Make a separate dataset for the registered image.
    logger.debug('new_size {}'.format(new_size))
    try:
        registered_image = im_file_n.require_dataset('reg_image',
                                             shape=tuple(new_size),
                                             dtype=np.float64)
    except TypeError as err:
        logger.debug(
            ("Incompatible data set for reg_image, deleting old " +
             "dataset. N.B: Not cleaning up space. \n {}")
            .format(err))
        del im_file_n['reg_image']
        registered_image = im_file_n.require_dataset('reg_image',
                                                 shape=tuple(new_size),
                                                 dtype=np.float64)

    # Transform the image
    registered_image[:, :] = smtf.warp(im_file_n['final_image'],
                                   trans_matrix,
                                   output_shape=new_size, order=0)


def transform_data_file(folder, data_name, trans,
                        new_size):
    """
    Transform the corners in the pickled data file

    Parameters:
    -----------

    folder: str
        The name of the folder containing the
        pickled file with stitching data, needs a
        trailing slash ("/").
    data_name: str
        Name of the pickled file with the corner coordinates.
    trans: np.array
        Array of len 2 containing ints. y and x transform of the image.
    new_size: tuple
        Tuple of length 2. The size of the image after the transform.
    """
    # Determine the name to safe the new pickled data file.
    exp_name = '_'.join(data_name.split('_')[:-2])

    # Get the original coordinates
    loaded_data = inout.load_stitching_coord(folder + data_name)
    micData = loaded_data['micData']
    joining_original = loaded_data['joining']
    joining_new = {}
    # Translate the corners
    temp_corner_list = [[tile_ind, (corner - trans)]
                        for tile_ind, corner in
                        joining_original['corner_list']]
    logger.debug(
        'temp_corner_list: {} trans: {}'.format(temp_corner_list,
                                                trans))
    # Place the corners in the joining dictionary.
    joining_new['corner_list'] = temp_corner_list
    # Change final image shape of original:
    joining_new['final_image_shape'] = new_size

    # Save to a new file
    inout.save_to_file(folder + exp_name + '_stitching_data_reg',
                       micData=micData, joining=joining_new)


def align_sub_region(overlap1, overlap2, nr_peaks):
    """Determine how much overlap2 should be shifted to fit overlap1.
    
    Parameters:
    -----------

    overlap1: np.array
        2D numpy array. Patch of the image that should be compared.
    overlap2: np.array
        2D numpy array. Patch of the image that should be compared.
    nr_peaks: int
        The number of peaks used to get the best peaks from the phase correlation matrix.
    
    Returns:
    --------

    best_trans: np.array
        Array of len 2 containing ints. Transform that projects overlap2 
        correctly onto overlap1.
    best_cov: float
        The normalized covariance
    """

    plot_order = np.ones((1, 2))
    # Calculate possible translations
    unr_pos_transistions = ps.calculate_pos_shifts(
        overlap1, overlap2, nr_peaks, 2)
    logger.debug("Possible translations: {}".
                 format(unr_pos_transistions))
    # Do correlation over the found shifts:
    best_trans, best_cov = ps.find_best_trans(unr_pos_transistions,
                                              overlap1, overlap2,
                                              plot_order)
    # Give some feedback
    logger.info(
        "Best shift: {} covariance: {}".format(best_trans, best_cov))
    return np.array(best_trans,dtype='int16'), best_cov


def register_final_images_old(folder, gene='Nuclei',
                              sub_pic_frac=0.2, use_MPI=False,
                              apply_to_corners=True,
                              apply_warping=False,
                              region=None, compare_in_seq=False):
    """Register stitched images an in all  HDF5 file in the folder

    Loops the hybridizations in the HDF5 file, takes the stitched
    images as indicated by gene and then compares each image to the
    first image.
    For the comparison only a small patch of the images is used, the
    size of this patch can be controlled with "sub_pic_frac".

    Parameters:
    -----------

    folder: str
        The name of the folder containing the
        pickled file with stitching data, needs a
        trailing slash ("/").
    gene: str
        The gene of which the stitched
        images are present and should be realigned.
        Typically this will be 'Nuclei', because the
        smFISH genes will not have enough signal to
        align the pictures properly.
        (Default: 'Nuclei')
    sub_pic_frac: float
        The fraction of the size of the
        original image that should be used to compare
        images.
        (Default: 0.2)
    use_MPI: bool
        True open the files in MPI friendly mode, if False open files in normal
        single processing mode. (Default: False)
    apply_to_corners: bool
        Determines if the found
        registration will be applied to the tile
        corners in the pickled stitching data file.
        (Default: True)
    apply_warping: bool
        Determines if the found
        registration will be applied as a warp to the
        final pictures in the hdf5 file, should not
        be used with large datasets.
        (Default: False)
    region: list
        List of length four containing ints. The
        region that should be compared to determine
        the shift needed for registration.
        Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac
        will not be used.
        By default the code will determine the region
        itself taking a area around the
        center of the image with a size
        determined by sub_pic_frac(Default: None)
    compare_in_seq: bool
        Determines if we should compare
        images in sequence or if we should compare
        all to the first image.

    """
    # Get a list of files in the folder
    file_name_list = glob.glob(folder + '*.sf.hdf5')
    file_name_list.sort()
    logger.debug('Filenames sorted: {}'.format(file_name_list))
    # Initialize some variables:
    trans = np.zeros((len(file_name_list), 2), dtype=int)
    old_size_list = np.zeros((len(file_name_list), 2), dtype=int)
    max_trans = np.zeros((1, 2), dtype=int)
    # Make comparisons
    if not compare_in_seq:
        # Take the first hybridization (keys() seems to give the groups
        # as a sorted list)
        im_name_1 = file_name_list[0]
        logger.debug('im_name_1: {}'.format(im_name_1))
        # Open the stitching file and make a list of the hybridizations
        # present in this file:
        if use_MPI:
            file_1 = h5py.File(im_name_1, 'r+',
                               driver='mpio', comm=MPI.COMM_WORLD)
        else:
            file_1 = h5py.File(im_name_1, 'r+')

        # hyb_name_list = list(stitching_file.keys())

        # Get the right group
        im_file_1 = file_1[gene]['StitchedImage']
        old_size_list[0, :] = im_file_1['final_image'].shape
        # Compare each file to file 1:
        for i in range(1, len(file_name_list)):
            # Get the group containing the image we want to compare with.
            if use_MPI:
                file_n = h5py.File(file_name_list[i], 'r+',
                                   driver='mpio', comm=MPI.COMM_WORLD)
            else:
                file_n = h5py.File(file_name_list[i], 'r+')
            im_file_n = file_n[gene]['StitchedImage']
            # Find the translation
            trans[i, :], max_trans, old_size_list[i, :] \
                = find_reg_final_image(im_file_1, im_file_n, max_trans,
                                       sub_pic_frac, region=region)
            logger.debug(
                "max_trans: {}".format(max_trans))
            file_n.close()
        # Close the hdf5 file.
        file_1.close()
    else:
        # Init specific array
        trans_relative = np.zeros((len(file_name_list), 2), dtype=int)
        # Compare each file to previous file:
        for i in range(1, len(file_name_list)):
            # Get the group containing the image we want to compare with.
            if use_MPI:
                file_1 = h5py.File(file_name_list[i - 1], 'r+',
                                   driver='mpio', comm=MPI.COMM_WORLD)
                file_2 = h5py.File(file_name_list[i], 'r+',
                                   driver='mpio', comm=MPI.COMM_WORLD)
            else:
                file_1 = h5py.File(file_name_list[i - 1], 'r+')
                file_2 = h5py.File(file_name_list[i], 'r+')
            im_file_1 = file_1[gene]['StitchedImage']
            im_file_2 = file_2[gene]['StitchedImage']
            # Get the size of the first image in the list,
            # which will be the reference image without translation.
            if (i - 1) == 0:
                old_size_list[0, :] = im_file_1['final_image'].shape
            # Find the translation
            trans_relative[i, :], max_trans, old_size_list[i, :] \
                = find_reg_final_image(im_file_1, im_file_2, max_trans,
                                       sub_pic_frac, region=region)
            logger.debug("max_trans: {}".format(max_trans))
            file_1.close()
            file_2.close()
        # Get the normalized transistions
        trans = np.cumsum(trans_relative, axis=0)
        max_trans = np.amax(trans, axis=0)
        logger.debug(("Comparing in sequence: relative translations: "
                      + "\n {} \n normalized translations: \n{}\n"
                      .format(trans_relative, trans)))
        logger.debug("max_trans: {}".format(max_trans))

    # Correct translations
    trans -= max_trans
    logger.debug('old_size_list: {}'
                 .format(old_size_list))
    # Determine final image size:
    new_size_list = old_size_list + abs(trans)
    new_size = np.amax(new_size_list, axis=0)
    logger.debug(
        'Files: {} Translations: {} new_size_list: {} new_size: {}'
            .format(file_name_list, trans, new_size_list, new_size))
    # Apply the translations
    for i in range(len(file_name_list)):
        if apply_warping:
            if use_MPI:
                file_n = h5py.File(file_name_list[i], 'r+',
                                   driver='mpio', comm=MPI.COMM_WORLD)
            else:
                file_n = h5py.File(file_name_list[i], 'r+')
            im_file_n = file_n[gene]['StitchedImage']
            transform_final_image(im_file_n, trans[i, :], new_size)
            file_n.close()
        if apply_to_corners:
            data_name = (
                os.path.split(file_name_list[i])[1].split(sep='.')[0]
                + '_' + gene
                + '_stitching_data')
            transform_data_file(folder, data_name, trans[i, :],
                                new_size)


def register_final_images_reg_data_only(folder, gene='Nuclei',
                          sub_pic_frac=0.2, use_MPI=False,
                          apply_to_corners=True, apply_warping = False,
                          region=None, compare_in_seq=False):
    """Register stitched images an in all  HDF5 file in the folders. 
    It is modified from register_final_images and saves only the reg_data
    file with the new coords and nothing in the hdf5 file.

    Loops the hybridizations in the HDF5 file, takes the stitched
    images as indicated by gene and then compares each image to the
    first image.
    For the comparison only a small patch of the images is used, the
    size of this patch can be controlled with "sub_pic_frac".

    Parameters:
    -----------

    folder: str
        The name of the folder containing the
        pickled file with stitching data, needs a
        trailing slash ("/").
    gene: str
        The gene of which the stitched
        images are present and should be realigned.
        Typically this will be 'Nuclei', because the
        smFISH genes will not have enough signal to
        align the pictures properly.
        (Default: 'Nuclei')
    sub_pic_frac: float
        The fraction of the size of the original image that should be used to compare
        images. (Default: 0.2)
    use_MPI: bool
        If True open the files in MPI friendly mode, if False open files in normal
        single processing mode. (Default: False)
    apply_to_corners: bool
        Determines if the found
        registration will be applied to the tile
        corners in the pickled stitching data file.
        (Default: True)
    apply_warping: bool
        Determines if the found
        registration will be applied as a warp to the
        final pictures in the hdf5 file, should not
        be used with large datasets.
        (Default: False)
    region: list
        List of length four containing ints. The
        region that should be compared to determine
        the shift needed for registration.
        Should be in the order: [y_min, y_max, x_min,
        x_max]. When region is defined, sub_pic_frac
        will not be used.
        By default the code will determine the region
        itself taking a area around the
        center of the image with a size
        determined by sub_pic_frac(Default: None)
    compare_in_seq: bool
        Determines if we should compare images in sequence or if we should compare
        all to the first image.

    """
    if not compare_in_seq:
        file_name_list, file_1, im_file_1, trans, old_size_list, \
            max_trans = \
            prepare_for_comparing(folder, gene, compare_in_seq,
                                  use_MPI=use_MPI)
        # Compare each file to file 1:
        for i in range(1, len(file_name_list)):
            cur_trans, max_trans, cur_old_size, file_ind = \
                get_single_trans(file_name_list, i, gene, im_file_1,
                                 max_trans, sub_pic_frac=sub_pic_frac,
                                 region=region, use_MPI=use_MPI)
            trans[file_ind, :] = cur_trans
            old_size_list[file_ind, :] = cur_old_size
        # Close the hdf5 file.
        file_1.close()
        trans, new_size = correct_trans_and_size(trans,
                                                 old_size_list,
                                                 max_trans,
                                                 compare_in_seq)
    else:
        file_name_list, trans_relative, old_size_list, max_trans = \
            prepare_for_comparing(folder, gene, compare_in_seq,
                                  use_MPI=use_MPI)
        # Compare each file to previous file:
        for i in range(1, len(file_name_list)):
            cur_trans, max_trans, cur_old_size, file_ind = \
                get_single_relative_trans(file_name_list, i, gene,
                                          max_trans,
                                          sub_pic_frac = sub_pic_frac,
                                          region = region,
                                          use_MPI = use_MPI)
            trans_relative[file_ind, :] = cur_trans
            old_size_list[file_ind, :]  = cur_old_size
        trans, new_size = correct_trans_and_size(trans_relative,
                                                 old_size_list,
                                                 max_trans,
                                                 compare_in_seq)

    logger.debug(
        'Files: {} Translations: {}'
            .format(file_name_list, trans))

    # Apply the translations
    for i in range(len(file_name_list)):
        if apply_to_corners:
            data_name = (
            os.path.split(file_name_list[i])[1].split(sep='.')[0]
            + '_' + gene
            + '_stitching_data')
            transform_data_file(folder, data_name, trans[i, :],
                                new_size)