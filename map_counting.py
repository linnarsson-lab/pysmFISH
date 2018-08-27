#!/usr/bin/env python

"""
Preprocessing script that run filtering and raw counting
"""
#
# Import dask distributed to monitor the running client
from distributed import Client,LocalCluster
from dask import delayed
from collections import OrderedDict
import os
import logging
import glob
from skimage import io, img_as_float,filters
import pickle
import argparse
import h5py
import yaml
import sys
import multiprocessing
import shutil
# -----------------
import numpy as  np



def map_counting():
    """
    This script is used to count the RNA molecules inside each segmented cells.
    The parameters are entered using argparse
    

    """


    # Inputs of the function
    parser = argparse.ArgumentParser(description='RNA mapping script')
    parser.add_argument('-scheduler', default=False, help='dask scheduler address ex. tcp://192.168.0.4:7003')
    parser.add_argument('-path', help='processing directory')
    args = parser.parse_args()
    
    # Directory to process
    processing_directory = args.path
    # Dask scheduler address
    scheduler_address = args.scheduler
    
    if scheduler_address:
        # Start dask client on server or cluster
        client=Client(scheduler_address)

    else:
        # Start dask client on local machine. It will use all the availabe
        # cores -1

        # number of core to use
        ncores = multiprocessing.cpu_count()-1
        cluster = LocalCluster(n_workers=ncores)
        client=Client(cluster)

    # Subdirectories of the processing_directory that need to be skipped for the
    # analysis
    blocked_directories = ['_logs']

    # Starting logger
    utils.init_file_logger(processing_directory)
    logger = logging.getLogger()

    # Determine the operating system running the code
    os_windows, add_slash = utils.determine_os()

    # Check training slash in the processing directory
    processing_directory=utils.check_trailing_slash(processing_directory,os_windows)

    # Get the list of the stitched files to process
    stitched_file_list = glob.glob(processing_directory + '*.sf.hdf5')


    # Load all the segmented object properties
    segmented_objs_fname = glob.glob(processing_directory + '*_all_objs_properties.pkl')[0]
    all_objects_properties_dict = pickle.load(open(segmented_objs_fname,'rb'))


    # Parse the configuration file 
    flt_rawcnt_config = utils.filtering_raw_counting_config_parser(processing_directory)

    use_ram = flt_rawcnt_config['use_ram']
    max_ram = flt_rawcnt_config['max_ram']
    analysis_name = flt_rawcnt_config['analysis_name']
    skip_genes_counting = flt_rawcnt_config['skip_genes_counting']


    # Loop through each hyb file in the folder
    for stitched_file_name in stitched_file_list:
        # extract the hyb processed
        hyb = stitched_file_name.split('.')[0].split('_')[-1]

        # load the file
        stitched_file_hdl = h5py.File(stitched_file,'r')

        # Create list of genes to process
        genes_to_keep=set(stitched_file_hdl)-set(skip_genes_counting)
            
        # process each gene
        for gene in genes_to_keep:

            size_counter = 0
            tmp_storage={}

            # Loop through the objects and load them in memory
             for obj_idx in all_objects_properties_dict.keys():
                obj_coords = all_objects_properties_dict[obj_idx]['obj_coords']
                row_Coords_obj = obj_coords[:,0]
                col_Coords_obj = obj_coords[:,1]

                # Calculate the coords of the bounding box to use to crop the ROI
                # Added +1 because in the cropping python will not consider the end of the interval
                bb_row_max=row_Coords_obj.max()+1
                bb_row_min=row_Coords_obj.min()
                bb_col_max=col_Coords_obj.max()+1
                bb_col_min=col_Coords_obj.min()

                # Normalize the coords of the obj
                row_Coords_obj_norm=row_Coords_obj-bb_row_min
                col_Coords_obj_norm=col_Coords_obj-bb_col_min


                bb_coords =  (bb_row_min,bb_row_max,bb_col_min,bb_col_max)
                obj_coords_norm = (row_Coords_obj_norm,col_Coords_obj_norm)

                # Load the cropped image
                Img_cropped=stitched_file_hdl[gene]XXXXXXXXXXXX['final_image'][bb_coords[0]:bb_coords[1],bb_coords[2]:bb_coords[3]] 

                if use_ram:
                    # Create a dict that is saved in ram. When full is written on disk
                    # Done to reduce the number of i/o and increase performance
                    size_counter += Img_cropped.nbytes
                    if size_counter < max_ram:
                        tmp_storage[obj_idx]={}
                        tmp_storage[obj_idx]['img'] = img_stack
                        tmp_storage[obj_idx]['bb_coords'] = bb_coords

                    else:
                        for obj_idx in tmp_storage.keys():
                            
                            # Scatter everything/count
    
                        tmp_storage={}
                        size_counter = Img_cropped.nbytes
                        tmp_storage[obj_idx]={}
                        tmp_storage[obj_idx]['img'] = img_stack
                        tmp_storage[obj_idx]['bb_coords'] = bb_coords 






    # Normalize the coords of the obj
    row_Coords_obj_norm=row_Coords_obj-bb_row_min
    col_Coords_obj_norm=col_Coords_obj-bb_col_min
    
    
    return (bb_row_min,bb_row_max,bb_col_min,bb_col_max),(row_Coords_obj_norm,col_Coords_obj_norm)





        for 





    if use_ram:
                    # Create a list that is saved in ram. When full is written on disk
                    # Done to reduce the number of i/o and increase performance
                    size_counter += img_stack.nbytes
                    if size_counter < max_ram:
                        tmp_storage[fov]={}
                        tmp_storage[fov]['img']=img_stack
                        tmp_storage[fov]['converted_fname']=converted_fname
                    else:
                        for pos in tmp_storage.keys():
                            np.save(tmp_storage[pos]['converted_fname'],tmp_storage[pos]['img'],allow_pickle=False)

                        tmp_storage={}
                        size_counter = img_stack.nbytes
                        tmp_storage[fov]={}
                        tmp_storage[fov]['img'] = img_stack
                        tmp_storage[fov]['converted_fname'] = converted_fname                  
                
                else:
                    # Directly save the file without saving it in RAM
                    np.save(converted_fname,img_stack,allow_pickle=False)




    # Get the list of the file to process with all the stitched images


# Load the dictionary with the info of the objects to process


# Load the configuration file with the RAM info to use for the quantification


# External loop through the files with stitched images


    # Loop on the cell selection to fill the memory 

        # Fill up the dictionary

        # Map the counting process

        # Reduce the result to a dictionary

    # When all the cells are processed collect all counts and store them on tmp hdf5











if __name__ == '__main__':
    map_counting()