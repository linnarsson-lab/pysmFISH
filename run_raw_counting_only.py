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
import sys
import multiprocessing
import shutil
# -----------------
import numpy as  np

from pysmFISH import utils,counting,dots_calling,io


def run_raw_counting_only():
    """
    This script will process all the hybridization folders combined in a 
    processing folder. The input parameters are passed using argparse

    Parameters:
    -----------
    
    scheduler: string
        tcp address of the dask.distributed scheduler (ex. tcp://192.168.0.4:7003). 
        default = False. If False the process will run on the local computer using nCPUs-1
    path: string
        Path to the processing directory
    counting_name: string
        String with an extra name to add to the folder/file name default = False.


    """
    # Inputs of the function
    parser = argparse.ArgumentParser(description='Preprocessing script')
    parser.add_argument('-scheduler', default=False, help='dask scheduler address ex. tcp://192.168.0.4:7003')
    parser.add_argument('-path', help='processing directory')
    parser.add_argument('-counting_name', help='extra name tag for counting')
    args = parser.parse_args()
    
    # Directory to process
    processing_directory = args.path
    # Dask scheduler address
    scheduler_address = args.scheduler
    # extra name tag
    counting_name = args.counting_name


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
    blocked_directories = ['_logs','naive','injured','stitched_reference_files']

    # Starting logger
    utils.init_file_logger(processing_directory)
    logger = logging.getLogger()

    # Determine the operating system running the code
    os_windows, add_slash = utils.determine_os()

    # Check training slash in the processing directory
    processing_directory=utils.check_trailing_slash(processing_directory,os_windows)

    # Get a list of the hybridization to process
    processing_hyb_list = next(os.walk(processing_directory))[1]

    # Remove the blocked directories from the directories to process
    processing_hyb_list = [el for el in processing_hyb_list if el not in blocked_directories ]

    for processing_hyb in processing_hyb_list:
    
        # Determine the hyb number from the name
        hybridization_number = processing_hyb.split('_hyb')[-1]
        hybridization = 'Hybridization' + hybridization_number
        hyb_dir = processing_directory + processing_hyb + add_slash
        
        # Parse the Experimental metadata file (serial)
        experiment_infos,image_properties, hybridizations_infos, \
        converted_positions, microscope_parameters =\
        utils.experimental_metadata_parser(hyb_dir)
        
        # Parse the configuration file 
        flt_rawcnt_config = utils.filtering_raw_counting_config_parser(hyb_dir)
        
    
        # Determine the directory of the filtered images
        suffix = 'filtered_npy'
        analysis_name=flt_rawcnt_config['analysis_name']
        sufx_dir_path = hyb_dir+analysis_name+'_'+processing_hyb+'_'+suffix+add_slash

        # Create the directory where to save the counting
        skip_genes_counting=flt_rawcnt_config['skip_genes_counting']
        skip_tags_counting=flt_rawcnt_config['skip_tags_counting']

        if counting_name:
            suffix = 'counting'
            counting_dir_path, counting_gene_dirs = \
                utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,processing_hyb,
                                suffix,add_slash,flt_rawcnt_config['skip_tags_counting'],
                                flt_rawcnt_config['skip_genes_counting'],
                                analysis_name=flt_rawcnt_config['analysis_name']+'_'+counting_name)
        else:
            suffix = 'counting'
            counting_dir_path, counting_gene_dirs = \
                utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,processing_hyb,
                                suffix,add_slash,flt_rawcnt_config['skip_tags_counting'],
                                flt_rawcnt_config['skip_genes_counting'],
                                analysis_name=flt_rawcnt_config['analysis_name'])
    

        # ----------------- RAW COUNTING ONLY------------------------    
        for gene in hybridizations_infos[hybridization].keys():

            # Filtering image according to gene
            if gene not in skip_genes_counting and [tag for tag in skip_tags_counting if tag not in gene]:
                suffix = 'filtered_npy'
                if analysis_name:
                    filtered_images_directory =  sufx_dir_path+analysis_name+'_'+processing_hyb+'_'+ gene+'_'+suffix+add_slash
                else:
                    filtered_images_directory =  sufx_dir_path +processing_hyb+'_'+ gene +'_'+suffix+add_slash
                
                flist_img_to_filter=glob.glob(filtered_images_directory+'*.npy')
                # counting
                logger.debug('counting %s',gene)

                futures_processes=client.map(counting.counting_only,flist_img_to_filter, \
                                        counting_gene_dirs=counting_gene_dirs, \
                                        min_distance=flt_rawcnt_config['min_distance'],\
                                        stringency=flt_rawcnt_config['stringency'])

                client.gather(futures_processes)


    client.close()
    

if __name__ == '__main__':
    run_raw_counting_only()