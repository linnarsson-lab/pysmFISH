# Module containing functions written for the in and output of tile microcope images
from skimage import io

# import matplotlib.pyplot as plt
# import visvis as vv
# plt_available = True
plt_available = False

import glob as glob
import pickle as pickle
import numpy as np
import logging
import csv
import ast
import re

def extract_hdf5_refs(hdf5_group, image_inds):
    """Make a list of the dataset is hdf5_group.

    When image_inds is not empty this function returns only the
    data sets corresponding to the numbers in image_inds.

    Parameters:
    -----------

    hdf5_group: pointer
        HDF5 group reference. Reference to the group
        from which we should extract the dataset names.
        The data sets are expected to be numbered.
    image_inds: iterable
        Iterable containing ints. Should contain the
        names of the datasets that will be selected. This
        list could also contain strings, as long as it's
        contents reflect a subset of dataset names.
    
    Returns:
    --------

    tile_refs: list
        List of strings. The references to the data sets.
    """
    # Get logger
    logger = logging.getLogger(__name__)
    # Get the full human readable reference to hdf5_group.
    group_root_name = hdf5_group.name + '/'
    if image_inds.size:
        # If image_inds is given match image_inds to the data sets
        tile_refs = []
        for im_nr in image_inds:
            logger.debug(
                "Trying the find image nr {} in hdf5 file".format(
                    im_nr))
            try:
                found_tile = hdf5_group[str(im_nr)]
                cur_ref = group_root_name + str(im_nr)
            except KeyError:
                cur_ref = None
            logger.debug("Appending reference: {}".format(cur_ref))
            tile_refs.append(cur_ref)
        return tile_refs
    else:
        # If no image_inds are given, return all data set references
        # in alphabetical order (relies on hdf5 alphabetical return
        # of keys)
        tile_refs = [group_root_name + tile_name for tile_name in
                     hdf5_group.keys()]
        logger.debug("Got all refs: {}".format(tile_refs))
        return tile_refs


def get_image_names(hdf5_file, image_list=np.array([]),
                    hyb_nr=1, gene='Nuclei', pre_proc_level='FilteredData'):
    """Get the references to the tiles from the hdf5 file.

    If no matching data set was found, None will be used
    instead of a string reference.
    
    Parameters:
    -----------

    hdf5_file: pointer
        HDF5 file handle. The file containing the tiles.
    image_list: iterable
        Iterable containing ints. Should contain the
        names of the data sets that will be selected. This
        list could also contain strings, as long as it's
        contents reflect a subset of dataset names.
    hyb_nr: int
        The number of the hybridization where the
        data sets can be found. This will be used to
        navigate tile_file and find the correct tiles.
        (Default: 1)
    gene: str
        The name of the gene where the data sets can be found.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Nuclei')
    pre_proc_level: str
        The name of the pre processing group
        where the data sets can be found.
        This will be used to navigate tile_file and find the
        correct tiles. (Default: 'Filtered')
    """
    #logger = logging.getLogger(__name__)
    #logger.info("Getting image references from hdf5 file...")
    # Get the right group
    cur_group = hdf5_file[gene][pre_proc_level]
    return extract_hdf5_refs(cur_group, image_list)


def load_tile(file_name, hdf5_file=None):
    """Load a tile into memory as 2D image.

    Parameters:
    -----------

    file_name: str 
        Hdf5 reference to tile data set.
    hdf5_file: pointer
        HDF5 file handle. Opened HDF5 file containing the
        tile.
    
    Returns:
    --------

    : np.array 
            The tile as 2D image. Or an empty array when the file_name is None.
    """
    #logger = logging.getLogger(__name__)
    #logger.info("Loading image from: {}".format(file_name))
    if file_name is not None:
        # Flatten the image
        # Flattening also causes the image to be copied from the
        # hdf5-file into working memory as numpy array. It is
        # important to have the image as a numpy array rather than a
        # hdf5 dataset, because we want to do fancy slicing, which
        # is not possible on a dataset.
        return np.amax(hdf5_file[file_name], axis=0)
    else:
        return np.array([])


def load_tile_3D(file_name, hdf5_file=None):
    """Load a tile into memory as 3D image.

    Parameters:
    -----------

    file_name: str
        Hdf5 reference to tile data set.
    hdf5_file: pointer
        HDF5 file handle. Opened HDF5 file containing the tile.
    
    Returns:
    --------

    : np.array
        The tile as 3D image. Or an empty array when the file_name is None.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading image from: {}".format(file_name))

    if file_name is not None:
            # Copy the image from the hdf5-file into working memory as
            # numpy array. It is important to have the image as a numpy
            # array rather than a hdf5 dataset, because we want to do
            # fancy slicing, which is not possible on a dataset.
            return hdf5_file[file_name][()]
    else:
        return np.array([])


def save_image(im_file, hyb_nr, gene, pre_proc_level, image_name,
               save_file_name):
    """
    Save hdf5 file as a tiff image

    Save the data found in im_file_name[hyb_nr][gene][pre_proc_level][
    image_name] as an image in a tiff-file called save_file_name.
    im_file should be an hdf5-file and image_name should be a
    variable in that file.
    
    Parameters:
    -----------

    im_file: pointer
        HDF5 file handle. Opened HDF5 file containing the image to be saved.
    hyb_nr: int 
        Number of the current hybridization.
    gene: str 
        The name of the gene where the
        image to be saved can be found.
        This will be used to navigate im_file and find the
        correct image.
    pre_proc_level: str
        The name of the pre processing group
        where the image to be saved can be found.
        This will be used to navigate im_file and find the
        correct image.
    image_name: str
        The name of the dataset containing the image to be saved.
    save_file_name: str
        The name of the image file that is going to be generated, should be given without
        extension.
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving image as: {}.tif".format(save_file_name))
    io.use_plugin('tifffile')
    #image_group = im_file['Hybridization' + str(hyb_nr)][gene][
    #    pre_proc_level]
    image_group = im_file[gene][pre_proc_level]

    io.imsave(save_file_name + ".tif", image_group[image_name],
              plugin="tifffile")


def save_to_file(file_name, **kwargs):
    """Save pickled stitching info to file_name

    Parameters:
    -----------

    file_name: str
        The name of the file where the data will be saved. Should be given without extension.
    
    **kwarg
        All keyword argument values will be
        gathered in a dictionary, with the variable name
        as their key in the dictionary. This dictionary will
        then be saved in the pickled file.

    """
    # Add extension
    file_name = file_name + '.pkl'
    logger = logging.getLogger(__name__)
    logger.info("Saving data to: {}".format(file_name))
    # Make a dictionary
    data = {}
    for key, val in kwargs.items():
        data[key] = val
    logger.debug("Saving data: {}".format(data))
    # Write to file
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_stitching_coord(file_name):
    """Load pickled variables

    Parameters:
    -----------

    file_name: str
        The name of a pickled data file. Should be given without extension.
    
    Returns:
    --------

    : dict  
        Unpickled data found in <file_name>'.pkl'
    """
    file_name = file_name + '.pkl'
    logger = logging.getLogger(__name__)
    logger.debug("Loading data from: {}".format(file_name))
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def free_hdf5_space(f):
    """Clean up the hdf5 file after deleting a dataset

    After deleting a data set from an hdf5-file, the disk space taken
    up by that data is not freed. Flushing the hfd5-file may help to
    keep the file smaller after deleting several data sets.

    Parameters:
    -----------

    f: pointer
        HDF5 file handle. The opened file that needs flushing.

    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Flushing hdf5 file to clean up after delete operations")
    before_flush = f.id.get_filesize()
    f.flush()
    after_flush = f.id.get_filesize()
    logger.debug(
        "Size in bytes before flush: {} after flush: {} space freed: {}".format(
            before_flush, after_flush, before_flush - after_flush))


def read_IJ_corners(folder):
    """Read Preibisch corners

    Read previously determined global corners from the Preibisch
    imageJ plugin data. This code starts reading below the line:
    '# Define the image coordinates' and the file should not contain
    empty lines after this line.

    Parameters:
    -----------

    folder: str  Exact path to the folder, including trailing "/"

    Returns:
    --------

    compare_corners: list
        List, each element is a list containing the
        tile_ind and a numpy arrray with the
        corresponding corner's y and x coordinates.

    """
    logger = logging.getLogger(__name__)
    # Key word to look for in the name of the  file with the previously calculated corners
    name_key = 'TileConfiguration'
    # Find the file, using the key word
    coord_file_name = next((name for name in glob.glob(folder + '*.txt')
                            if name_key in name), None)
    logger.info(
        "Reading corners from folder: {} and file: {}".format(folder,
                                                              coord_file_name))
    with open(coord_file_name, 'r') as coord_file:
        coord_reader = csv.reader(coord_file, delimiter=';')
        start_reading = False
        # Init corner list
        compare_corners = []
        for line in coord_reader:
            logger.debug(line)
            if start_reading:
                # Take the last number found in the string as number of the tile
                tile_nr = int(re.findall('\d+', line[0])[-1])
                # Convert corner coordinates
                corner = ast.literal_eval(line[-1].strip())
                # Append the data we want to use:
                compare_corners.append([tile_nr, corner])
                logger.debug('{} {}'.format(tile_nr, corner))
            # Read from '# Define the image coordinates' till the end
            if line == ['# Define the image coordinates']:
                # Found starting line
                logger.debug('starting to read')
                start_reading = True
    return compare_corners


######################## Plotting ######################################
def display_tiles(tiles, tile_set, fig_nr=1, block=True,
                  maximize=False, main_title=None, rgb=False):
    """Plot tiles in subplots

    Parameters:
    -----------

    tiles: list
        List of tiles
    tile_set: np.array
        Tiles organization
    fig_nr: int     
        number of the figure window.
    block: bool
        Block causes the program to stop running after plotting the figure,
        until the figure window is closed.
    maximize: bool 
        Maximize plots the figure full screen.
    main_title: str
        Main title adds a title text in the figure.
    rgb: bool
        define the color standard used
    """
    if plt_available:
        logger = logging.getLogger(__name__)
        logger.info("Plotting tiles")
        plt.figure(fig_nr)
        for i in range(len(tiles)):
            plt.subplot(tile_set.shape[0], tile_set.shape[1], i + 1)
            if tiles[i].ndim > 2 and not (rgb):
                plt.imshow(tiles[i][0, :, :], 'gray',
                           interpolation='none')
            else:
                plt.imshow(tiles[i], 'gray', interpolation='none')
        if maximize:
            logger.debug(
                "Matplotlib backend: {}".format(plt.get_backend()))
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if main_title:
            plt.suptitle(main_title, fontsize=20)
        plt.show(block=block)
    else:
        return None


def plot_stitched_coordinates(corner_list, image_list, block=True,
                              name='stitched_coordinates',
                              file_name=None,
                              invert_yaxis=True):
    """Plot top-left coordinates with their tile numbers

    Coordinates are assumed to be in pixels.
    
    Parameters:
    -----------
    corner_list : list
        List with as many elements as there are tiles. Each element contains
        the tile index (as int) and y and x coordinate of the top left corner
        (as a numpy array)
    image_list: list
        The list of the numbers of the tiles as they are found in the file names,
        but reordered to match the order of corner_list.
    block: bool
        Plot blocks the running program untill the plotting window is closed if true
        (default: True)
    name: str 
        Name of the plotting window (default: "coordinates")
    file_name: str
        Name of the file the image should be saved to.
        The name should list the complete absolute map.
        Default/None causes the image to not be saved. (Default: None)
    invert_yaxis: bool
        Invert the y-axis when plotting. (default: True)
    """
    if plt_available:
        labels = [str(corner_list[i][0]) + '/' + str(image_list[i])
                  for i in range(len(corner_list))]
        corners = np.array([corner[1] for corner in corner_list])
        fig = plt.figure(name)
        plt.plot(corners[:, 1], corners[:, 0], 'o')
        # plt.setp(plt.gca(), 'ylim', reversed(plt.getp(plt.gca(), 'ylim')))
        # bboxes = []
        # ax = plt.gca()
        for label, x, y in zip(labels, corners[:, 1], corners[:, 0]):
            annotation = plt.annotate(
                label,
                xy=(x, y), xytext=(-2, 2),
                textcoords='offset points', ha='center', va='bottom')
            # plt.draw()
            # fig.canvas.draw()
            # bbox = annotation.get_window_extent()
            # the figure transform goes from relative coords->pixels and we
            # want the inverse of that
            # bbox_data = bbox.transformed(ax.transData.inverted())
            # bboxes.append(bbox_data)
        # Title and axes labels
        plt.title(
            'Labels represent: <tile index>/<image number as it appears in file name>',
            y=1.08)
        plt.xlabel('top-left position (pxls)')
        plt.ylabel('top-left position (pxls)')

        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim(
            (ymin - 1000, ymax + 1000))  # set the ylim to ymin, ymax
        xmin, xmax = plt.xlim()  # return the current ylim
        plt.xlim(
            (xmin - 1000, xmax + 1000))  # set the ylim to ymin, ymax

        # bbox = mtransforms.Bbox.union(bboxes)
        # ax.update_datalim(bbox.corners())
        # ax.autoscale_view()

        if invert_yaxis:
            ax = plt.gca()
            plt.setp(ax, 'ylim', reversed(plt.getp(plt.gca(), 'ylim')))
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
        plt.tight_layout()
        # Make sure the labels are within the axes
        # plt.draw()
        # bboxes = []
        # for bbox in bboxes:
        # bbox = annotation.get_window_extent()
        # the figure transform goes from relative coords->pixels and we
        # want the inverse of that
        # bbox_data = bbox.transformed(ax.transData.inverted())
        # bboxes.append(bbox_data)



        # Save and/or display
        if file_name is not None:
            plt.savefig(file_name + '.png')
        plt.show(block=block)
    else:
        return None


def plot_coordinates(micData, ind_coord_list=None, block=True,
                     name="coordinates",
                     file_name=None, invert_yaxis=True):
    """Plot top-left coordinates with their tile numbers

    Coordinates are assumed to be in pixels.
    
    Parameters:
    -----------
    
    micData: object
        MicroscopeData object. Containing the coordinates to be plotted.
    block: bool
        Plot blocks the running program untill the plotting window is closed if true
        (default: True)
    name: str
        Name of the plotting window (default: "coordinates")
    file_name: str
        Name of the file the image should be saved to.
        The name should list the complete absolute map.
        Default/None causes the image to not be saved. (Default: None)
    invert_yaxis: bool
        Invert the y-axis when plotting. (default: True)
    """
    if plt_available:
        # Define labels
        if ind_coord_list is None:
            labels = [str(nr) for nr in micData.tile_nr]
        else:
            labels = [str(ind_coord_list[i][0]) + '/' + str(
                ind_coord_list[i][1])
                      for i in range(len(ind_coord_list))]
        plt.figure(name)
        plt.plot(micData.x_coords, micData.y_coords, 'o--')
        # Add annotations
        for label, x, y in zip(labels, micData.x_coords,
                               micData.y_coords):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-2, 2),
                textcoords='offset points', ha='center', va='bottom')
        # Axes labels
        plt.xlabel('top-left position (pxls)')
        plt.ylabel('top-left position (pxls)')
        # Enlarge axes so the annotation will fit within the axis better
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim(
            (ymin - 1000, ymax + 1000))  # set the ylim to ymin, ymax
        xmin, xmax = plt.xlim()  # return the current ylim
        plt.xlim(
            (xmin - 1000, xmax + 1000))  # set the ylim to ymin, ymax

        if invert_yaxis:
            ax = plt.gca()
            plt.setp(ax, 'ylim', reversed(plt.getp(plt.gca(), 'ylim')))
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

        # Save and/or display
        if file_name is not None:
            plt.savefig(file_name + '.png')
        plt.show(block=block)
    else:
        return None


def plot_3D(image):
    if plt_available:
        logger = logging.getLogger(__name__)
        logger.info("Plotting tiles in 3D")
        logger.debug("Shape pic: {}".format(image.shape))
        plt.figure('test final img')
        plt.imshow(image[0, :, :])
        plt.show()

        np_image = np.array(image)
        app = vv.use()
        vv.volshow(np_image, renderStyle='mip')
        a = vv.gca()
        a.camera.fov = 45
        a.daspectAuto = False
        a.daspect = (1, 1, 10)
        app.Run()
    else:
        return None
