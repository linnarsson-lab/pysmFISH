import argparse
import pickle
import numpy as np
import glob
import nd2reader as nd2
import os
import logging
import shutil

from distributed import Client
from skimage import io

from pysmFISH.counting import filtering_and_counting_experiment
from pysmFISH import utils

def process_standalone_experiment():

    """
    Script to run conversion, filtering and raw counting on a small set of images.
    The analysis run locally

    All the parameters are entered with argparse

    Parameters:
    -----------

    path: string
        Path to the experiment to process
    analysis_name: string
        Name of the analysis
    stringency: int
        Value of the stringency to use in the threshold selection. Default=0
    min_distance: int
        Min distance betwenn to peaks. Default=5
    min_plane: int
        Min plane for z-stack cropping. Default=None
    max_plane: int:
        Max plane for z-stack cropping. Default=None
    ncores: int
        Number of cores to use for the processing. Deafault=1


    """

    # input to the function
    parser = argparse.ArgumentParser(description='Counting and filtering experiment')
    parser.add_argument('-path', help='path to experiment to analyze')
    parser.add_argument('-analysis_name', help='analysis name')
    parser.add_argument('-stringency', help='stringency', default=0, type=int)
    parser.add_argument('-min_distance', help='min distance between peaks', default=5, type=int)
    parser.add_argument('-min_plane', help='starting plane to consider', default=None, type=int)
    parser.add_argument('-max_plane', help='ending plane to consider', default=None, type=int)
    parser.add_argument('-ncores', help='number of cores to use', default=1, type=int)



    # Parse the input args
    args = parser.parse_args()
    processing_directory = args.path
    analysis_name = args.analysis_name
    stringency = args.stringency
    min_distance = args.min_distance
    min_plane = args.min_plane
    max_plane = args.max_plane
    ncores  = args.ncores
                  

    if min_plane != None and max_plane != None:
        plane_keep = [min_plane, max_plane]
    else:
        plane_keep = None


    # Determine the os type
    os_windows, add_slash = utils.determine_os()

    # Starting logger
    utils.init_file_logger(processing_directory)
    logger = logging.getLogger()

    logger.debug('min_plane%s', min_plane)
    logger.debug('max_plane %s', max_plane)
    logger.debug('keep_planes value %s',plane_keep)

    # Start the distributed client
    client = Client(n_workers = ncores, threads_per_worker=1)

    logger.debug('client %s',client)
    logger.debug('check that workers are on the same directory %s', client.run(os.getcwd))
    
    # Check trail slash
    processing_directory = utils.check_trailing_slash(processing_directory,os_windows)


    # Determine the experiment name
    exp_name = processing_directory.split(add_slash)[-2]

    logger.debug('Experiment name: %s', exp_name)


    # Create the directories where to save the output
    tmp_dir_path = processing_directory+analysis_name+'_'+exp_name+'_tmp'+add_slash
    filtered_dir_path = processing_directory+analysis_name+'_'+exp_name+'_filtered'+add_slash
    counting_dir_path = processing_directory+analysis_name+'_'+exp_name+'_counting_pkl'+add_slash
    try:
        os.stat(tmp_dir_path)
    except:
        os.mkdir(tmp_dir_path)
        os.chmod(tmp_dir_path,0o777)

    try:
        os.stat(filtered_dir_path)
    except:
        os.mkdir(filtered_dir_path)
        os.chmod(filtered_dir_path,0o777)

    try:
        os.stat(counting_dir_path)
    except:
        os.mkdir(counting_dir_path)
        os.chmod(counting_dir_path,0o777)


    # Get the list of the nd2 files to process inside the directory
    files_list = glob.glob(processing_directory+'*.nd2')
    logger.debug('files to process %s', files_list)


    # Convert the .nd2 data
    for raw_data_gene_fname in files_list:
        fname = raw_data_gene_fname.split(add_slash)[-1][:-4]
        logger.debug('fname %s', fname)
        with nd2.Nd2(raw_data_gene_fname) as nd2file:
            for channel in nd2file.channels:
                for fov in nd2file.fields_of_view:
                    img_stack = np.empty([len(nd2file.z_levels),nd2file.height,nd2file.width],dtype='uint16')
                    images =  nd2file.select(channels=channel, fields_of_view=fov,z_levels=nd2file.z_levels)
                    for idx,im in enumerate(images):
                        img_stack[idx,:,:] = im

                    converted_fname=tmp_dir_path+exp_name+'_'+fname+'_'+channel+'_fov_'+str(fov)+'.npy'
                    np.save(converted_fname,img_stack,allow_pickle=False)     


    logger.debug('Finished .nd2 file conversion')

    # Filtering all the data
    # Get list of the files to process
    flist_img_to_filter=glob.glob(tmp_dir_path+'*.npy')

    # logger.debug('files to filter %s',flist_img_to_filter)
    # Parallel process all the data
    futures_processes=client.map(filtering_and_counting_experiment,flist_img_to_filter, \
                                  filtered_dir_path=filtered_dir_path, \
                                 counting_dir_path=counting_dir_path, \
                                 exp_name=exp_name,plane_keep=plane_keep,add_slash=add_slash, \
                                 min_distance=min_distance, stringency=stringency)
    
    
    client.gather(futures_processes)               
    client.close()

    logger.debug('Finished filtering and counting')         

    # delete the tmp folders
    shutil.rmtree(tmp_dir_path)

if __name__ == "__main__":
    process_standalone_experiment()