"""
Collection of utility functions.

"""

import os
import logging
from logging.handlers import RotatingFileHandler
import glob
import time
import yaml
import ruamel.yaml
import codecs
from collections import OrderedDict
import nd2reader as nd2
import numpy as np
from skimage import img_as_float
import platform


##################################Logging###############################
"""
The following lines will be executed when this file is imported, this code
initializes a logger, but does not use the logger. To print the output from
the logger run the function init_file_logger and/or init_console_logger, to save
output to a file or send the output to the stderr stream respectively.
"""
# Redirect warnings to logger
logging.captureWarnings(True)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def init_file_logger(log_path, maxBytes=0, backupCount=0):
    """
    Send the logging output to a file.

    The logs are placed in a directory called "logs", if this directory
    is not found at the location denoted by log_path, it will be made.
    On each run of the program a new log file will be produced.
    max_bytes and backup_count are passed to RotatingFileHandler
    directly. So a new file will be produced when the filesize
    reaches max_bytes. If either of max_bytes or backup_count is zero,
    rollover never occurs and everything will be logged in one file.
    
    Parameters:
    -----------

    log_path: str 
        Full path to the directory where the log directory is/should go.
    maxBytes: int 
        Maximum size in bytes of a single log file, a new log file will be 
        started when this size is reached. If zero, all logging will be 
        written to one file. (Default: 0)
    backup_count: int 
        The maxinum number of logging file that will be produced. If zero, 
        all logging will be written to one file. (Default: 0)
    """
    formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
    log_path += '_logs/'
    # To print the logging to a file
    try:
        os.stat(log_path)
    except:
        os.mkdir(log_path)
        os.chmod(log_path, 0o777)

    date_tag = time.strftime("%y%m%d_%H_%M_%S")
    fh = RotatingFileHandler(log_path +'%s-smFISH_Analysis.log' % date_tag,
                            mode ='w', maxBytes=maxBytes,
                            backupCount=backupCount)

    # Set logger properties
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    logger.info("Created file logger")


def init_console_logger():
    """
    Send the logging output to the stderr stream.
    After running this function the logging message will typically end up
    in your console output.
    """
    formatter_con = logging.Formatter('%(name)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_con)
    logger.addHandler(ch)
    logger.info("Created console logger")

########################################################################

def experimental_metadata_parser(hyb_dir):
    """
    Parse the yaml file containing all the metadata of the experiment

    The file must be located inside the experimental folder.

    Parameters:
    -----------

    hyb_dir: str
        Path to the .yaml file containing the metadata of the experiment

    Returns:
    -----------

    experiment_infos: dict 
        Dictionary with the information on the experiment.
    HybridizationInfos: dict 
        Dictionary with the information on the hybridization.
    converted_positions: dict 
        Dictionary with the coords of the images for all hybridization. 
        The coords are a list of floats
    microscope_parameters: dict 
        Dictionary with the microscope parameters for all hybridization

    """
    metadata_file_name = hyb_dir+'Experimental_metadata.yaml'
    logger.info("Parsing metadata from file: {}"
                .format(metadata_file_name))
    with open(metadata_file_name, 'r') as stream:
        try:
            docs=yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    experiment_infos = docs['ExperimentInfos']

    image_properties = docs['ImageProperties']
    hybridizations_infos = docs['HybridizationsInfos']
    Positions = docs['TilesPositions']
    microscope_parameters = docs['MicroscopeParameters']

    # Dictionary that will contain the coords after been coverted to float
    converted_positions={}
    # Convert the positions from list of string to list of floats
    for hyb, coords_dict in Positions.items():
        sub_dict = {}
        for pos, coords in coords_dict.items():
            coords = [float(x) for x in coords.split(',')]
            sub_dict[pos] = coords
        converted_positions[hyb] = sub_dict

    return experiment_infos,image_properties, hybridizations_infos,\
           converted_positions, microscope_parameters



def filtering_raw_counting_config_parser(hyb_dir):
    """
    Parse the yaml file containing all configurations for running the analysis

    The file must be located inside the experimental folder.

    Parameters:
    -----------

    hyb_dir: str 
        Path to the .yaml file containing the metadata of the experiment

    Returns:
    -----------

    config_parameters: dict 
        Dictionary with all the configuration parameters
    """
    configuration_file_name = hyb_dir+'Filtering_raw_counting.config.yaml'
    logger.info("Parsing metadata from file: {}"
            .format(configuration_file_name))
    with open(configuration_file_name, 'r') as stream:
        try:
            config_parameters=yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config_parameters


def general_yaml_parser(file_path):
    """
    Parse a general yaml file and return the dictionary with all the 
    content

    The file must be located inside the experimental folder.

    Parameters:
    -----------

    file_path: str 
        Path to the .yaml file containing the metadata of the experiment

    Returns:
    -----------

    parameters: dict 
        Dictionary with all the configuration parameters
    """
    
    logger.info("Parsing metadata from file: {}"
            .format(file_path))
    with open(file_path, 'r') as stream:
        try:
            parameters=yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return parameters


def determine_os():
    """
    This function check if the system is running windows.
    and return the correct slash type to use

    Returns:
    --------

    os_windows: bool 
        True if the os is windows.
    add_slash: str 
        '\' for windows or '/' for any other system

    """

    if 'Windows' in platform.platform():
        os_windows = True
        add_slash = '\\'
    else:
        os_windows = False
        add_slash = '/'

    if os_windows:
        logger.debug('OS: Windows')
    else:
        logger.debug('OS: Linux based')
        
    return os_windows, add_slash



def check_trailing_slash(dir_path, os_windows):
    """
    This function check if there is a trailing slash at the end of a 
    directory path and add it if missing

    Paramenters:
    ------------

    dir_path= str 
        Path to the directory

    Returns:
    --------

    dir_path= str 
        Path to the directory

    """
    logger.info('Checking the trailing slash ')
    
    if os_windows:
        if dir_path[-1]=='\\':
            logger.info('trailing slash present')
        else:
            logger.info('missing trailer slash, added now')
            dir_path=dir_path+'\\'
    else:

        if dir_path[-1]=='/':
            logger.info('trailing slash present')
        else:
            logger.info('missing trailer slash, added now')
            dir_path=dir_path+'/'
    return dir_path


def create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,processing_hyb,suffix,add_slash,
                                skip_tags=None,skip_genes=None,analysis_name=None):
    """
    Function that creates the directory tree where to save the
    temporary data.
    
    Parameters:
    -----------

    hyb_dir: str
        Path of the hyb to process
    hybridization: str 
        Name of the hybridization to process (ex. Hybridization2)
    hybridizations_infos: dict 
        Dictionary containing the hybridizations info parsed from the 
        Experimental_metadata.yaml file
    processing_hyb: str 
        Name of the processing experiment (ex. EXP-17-BP3597_hyb2)
    suffix: str 
        Suffix to add to the folder with useful description (ex. tmp)
    add_slash: str 
        '\\' for win and '/' for linux
    skip_tags: list 
        tags that won't be processed (ex. _IF)
    skip_genes list
         list of genes to skip
    analysis_name: str 
        Name of the analysis run

    Returns:
    ---------

    sufx_dir_path: str
        Path of the sufx directory of the processed hybridization
    sufx_gene_dirs: list 
        List of the paths of the sufx directory for the genes to process
    """

    logger.info('create {} directory'.format(suffix))

    gene_list = list(hybridizations_infos[hybridization].keys())
    # logger.debug('gene list: {}'.format(gene_list))
    sufx_gene_dirs = []
    if analysis_name: 
        # Create sufx directory
        sufx_dir_path = hyb_dir+analysis_name+'_'+processing_hyb+'_'+suffix+add_slash
        logger.debug('create {} directory'.format(suffix))
    else:
        # Create sufx directory
        sufx_dir_path = hyb_dir+processing_hyb+'_'+suffix+add_slash
    try:
        os.stat(sufx_dir_path)
    except:
        os.mkdir(sufx_dir_path)
        os.chmod(sufx_dir_path,0o777)
    
    
    if skip_genes:
        gene_list = [gene for gene in gene_list if gene not in skip_genes]
        
    if skip_tags:
        gene_list = [gene for tag in skip_tags for gene in gene_list if tag not in gene]

    for gene in gene_list:         
        if analysis_name:
            sufx_gene_dir_path = sufx_dir_path+analysis_name+'_'+processing_hyb+'_'+ gene+'_'+suffix+add_slash
            sufx_gene_dirs.append(sufx_gene_dir_path)
        else:
            sufx_gene_dir_path = sufx_dir_path +processing_hyb+'_'+ gene +'_'+suffix+add_slash
            sufx_gene_dirs.append(sufx_gene_dir_path)
        try:
            os.stat(sufx_gene_dir_path)
        except:
            os.mkdir(sufx_gene_dir_path)
            os.chmod(sufx_gene_dir_path,0o777)
    return sufx_dir_path, sufx_gene_dirs


def identify_nodes(client):
    """
    Function used to determine the address of the nodes in order to 
    better split the work

    Parameters:
    -----------

    client: dask.obj 
        Dask.distributed client.
    
    Returns:
    -----------

    node_addresses: OrderedDict
        Ordered dictionary. The keys are the addresses of the nodes and the 
        items are the full addresses of teh workers of a specific node.
    
    """
    logger.info('Determine the tcp addresses of the workers')
    
    # Gather the addresse of all the instantiated workers
    client_infos = client.scheduler_info()
    workers_addresses = client_infos['workers'].keys()
    
    # Isolate the tcp address of the nodes
    nodes_addresses = OrderedDict()
    nodes_comparison_list = []
    for address in workers_addresses:
        address_split = address.split(':')
        node_address = address_split[1].split('//')[-1]
        final_digits = address_split[-1]   
        if node_address in nodes_comparison_list:
            nodes_addresses['tcp://'+node_address][final_digits]=address
        else:
            nodes_comparison_list.append(node_address)
            nodes_addresses['tcp://'+node_address]={}
            nodes_addresses['tcp://'+node_address][final_digits]=address
    
    return nodes_addresses


def combine_gene_pos(hybridizations_infos,converted_positions,hybridization):
  

    """
    Gather info about the imaging at each hybridization.

    This function creates a dictionary where for each hybridization
    are shown the genes and number of positions imaged for each
    gene. This function will be useful to created distribution lists
    for running parallel processing of the datasets.

    Parameters:
    -----------

    hybridizations_infos: dict
        Dictionary with parsed Hybridizations metadata
    converted_positions: dict 
        Dictionary with the coords of the images for all hybridization. 
        The coords are a list of floats
    hybridization: str
        Selected hybridization to process

    Returns:
    -----------

    genes_and_positions: dict 
        Dictionary where for each hybridization, the genes and number of 
        positions imaged for each gene are showed.
    """
    
    genes_and_positions=dict()

    for gene in hybridizations_infos[hybridization].keys():
            genes_and_positions[gene] = list(converted_positions[hybridization].keys())
    
    return genes_and_positions


def partial_image_mean(img_paths):

    """
    Helper function used to calculate the mean of a set of images. It runs on a
    worker and help parallel image processing
    
    Parameters:
    -----------

    img_paths: list 
        List of paths to the images saved as *.npy


    Returns:
    -----------

    ImgMean: np.array 
        Array storing the calculated image mean

    """

    ImgMean = None
    for img_path in img_paths:
        img_stack = np.load(img_path)
        img_stack = img_as_float(img_stack)
        if ImgMean is None:
                ImgMean = img_stack
        else:
            ImgMean = (ImgMean + img_stack)/2.0
    return ImgMean

def list_chunking(list_to_chunk,num_chunks):

    """
    Helper function used to chunk a list in a number of sublists equal to
    num_chunks

    Parameters:
    -----------

    list_to_chunk: list 
        List to be chunked
    num_chunks: int 
        Number of sublists to obtain


    Returns:
    -----------

    chunked_list: list 
        List containing the chunked lists
    """
    size = np.int(len(list_to_chunk)/num_chunks)
    chunked_list = [list_to_chunk[i:i+size] for i  in range(0, len(list_to_chunk), size)]
    # if (len(list_to_chunk) - size*num_chunks)>0:
    #     chunked_list[-2].append(chunked_list[-1])
    #     del chunked_list[-1]
    return chunked_list


def create_single_directory(hyb_dir,gene,hybridization,processing_hyb,suffix,add_slash,
                                analysis_name=None):
    
    """
    Function used to create a subdirectory
    
    Parameters:
    -----------
    
    hyb_dir: str  
        Path to the directory of the hybridization currently processed.
    gene: str 
        Gene name to be included in the directory.
    processing_hyb: str 
        Name of the hybridization processed (ex. 'EXP-17-BP3597_hyb2').
    suffix: str 
        Extra info to add to the directory name (ex. blended).
    add_slash: str 
        Slash added according to the os.
    analysis_name: str 
        Name of the analysis associated to the folder
    
    Return
    ---------

    sufx_dir_path: str 
        Path to the created directory
    
    """
    
    logger.info('create {} directory'.format(suffix))
   
    sufx_gene_dirs = []
    if analysis_name: 
        # Create sufx directory
        sufx_dir_path = hyb_dir+analysis_name+'_'+processing_hyb+'_'+gene+'_'+suffix+add_slash
        logger.debug('create {} directory'.format(suffix))
    else:
        # Create sufx directory
        sufx_dir_path = hyb_dir+processing_hyb+'_'+gene+'_'+suffix+add_slash
    try:
        os.stat(sufx_dir_path)
    except:
        os.mkdir(sufx_dir_path)
        os.chmod(sufx_dir_path,0o777)
    
    return sufx_dir_path


def add_coords_to_yaml(folder, hyb_nr, hyb_key ='Hyb'):
    """
    Read tile number and coordinates and add them to the yaml file.
    Read the tile number and microscope coordinates for each tile from
    the microscope file called "coord_file_name" in "folder".
    Then insert them in dictionary "TilesPositions" in the yaml
    metadata file called Experimental_metadata.yaml.
    
    Parameters:
    -----------
    folder: str
        Exact path to the folder, including trailing "/"
    hyb_nr: int
        Hybridization number denoting for which hybridization we should read 
        and insert the coordinates
    hyb_key: str 
        Possible values 'Hyb' or 'Strip'. To add coordinates for stripping 
        if necessary.
    """

    # Key word to look for in the name of the coordinate file
    name_key    = 'Coords'
    hyb_key_filename     = hyb_key + str(hyb_nr)
    # Find the right file using "name_key" and hyb key
    coord_file_name = next((name for name in glob.glob(folder + '*.txt')
                           if (name_key in os.path.basename(name)) and
                           (hyb_key_filename in os.path.basename(name)))
                           , None)
    logger.info("Reading coordinates from file: {}"
                 .format(coord_file_name))

    if coord_file_name is None:
        logger.error("Coordinate file not found in folder {}, "
                      + "looking for txt-file including {} and {} in "
                      + "its name."
                      .format(folder, name_key, hyb_key_filename))

    # Load the yaml file with a special roundtrip loader, to change
    # it while keeping all comments and indentation
    metadata_file_name= folder + 'Experimental_metadata.yaml'
    with open(metadata_file_name, 'r') as stream:
        meta_data = ruamel.yaml.load(stream,
                                     ruamel.yaml.RoundTripLoader)
    stream.close()

    # Put coordinates from microscope file in TilesPositions
    if hyb_key == 'Hyb':
        cur_positions = meta_data['TilesPositions']['Hybridization'+ str(hyb_nr)]
    elif hyb_key == 'Strip':
        cur_positions = meta_data['TilesPositions']['Stripping'
                                                    + str(hyb_nr)]
    else:
        logger.warning("hyb_key not recognized, possible values are: "
                       + "'Hyb' or 'Strip'. Current value is: {}"
                       .format(hyb_key))

    # Open the coordinate file
    with codecs.open(coord_file_name, 'r', 'utf-16') as coord_file:
        # Use all lines starting with "#", assume these contain, index,
        # x and y coordinates
        for line in coord_file:
            # Use all lines starting with "#", that do not contain info
            # about "DAPI", assume these contain, index, x and y
            # coordinates
            if ('#' in line):
                # Replace the commas with dots, in case Windows
                # forced their Swedish commas on everything.
                # Clean up and split the line:
                replace_dict = {ord('#'): None, ord(','): ord('.')}
                line = line.translate(replace_dict).split('\t')
                logger.debug("Line read from coord file: {}"
                             .format(line))

                sep = ', '
                # Append the data we want to use:
                index   = int(line[0]) - 1
                x_value = float(line[1])
                y_value = float(line[2])
                z_value = float(line[3])

                cur_positions[index] = str(x_value) \
                                       + sep + str(y_value) \
                                       + sep + str(z_value)
            # Get out of the loop after we got to "Spectral Loop",
            # because we are not interested in the info that comes
            # after that.
            if ("Spectral Loop:" in line):
                break
    coord_file.close()

    # Place everything back into the file.
    with open(metadata_file_name, 'w') as stream:
            ruamel.yaml.dump(meta_data, stream,
                             Dumper=ruamel.yaml.RoundTripDumper,
                             default_flow_style=True, indent=4,
                             canonical=False)
    stream.close()
