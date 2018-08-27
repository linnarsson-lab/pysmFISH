#!/usr/bin/env python
# Import dask distributed to monitor the running client
from distributed import Client,LocalCluster
from dask import delayed
from collections import OrderedDict
import os
import logging
import glob
import nd2reader as nd2
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

from pysmFISH import utils,counting,dots_calling,io, hdf5_utils

from pysmFISH.stitching_package import stitching
from pysmFISH.stitching_package import tilejoining
from pysmFISH.stitching_package import hdf5preparation
from pysmFISH.stitching_package import pairwisesingle


def filtering_speed():
    """
    This script will process all the hybridization folders combined in a 
    processing folder. The input parameters are passed using arparse

    Parameters:
    -----------
    
    scheduler: string
        tcp address of the dask.distributed scheduler (ex. tcp://192.168.0.4:7003). 
        default = False. If False the process will run on the local computer using nCPUs-1

    path: string
        Path to the processing directory


    """


    # Inputs of the function
    parser = argparse.ArgumentParser(description='Preprocessing script')
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
        
    
        
        # ----------------- FILTERING AND RAW COUNTING ------------------------
        
        # Create directories 

        # Create the directory where to save the filtered images
        suffix = 'filtered_png'
        filtered_png_img_dir_path, filtered_png_img_gene_dirs = \
                utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,
                            processing_hyb,suffix,add_slash,analysis_name=flt_rawcnt_config['analysis_name'])

        suffix = 'filtered_npy'
        filtered_img_dir_path, filtered_img_gene_dirs = \
                utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,
                            processing_hyb,suffix,add_slash,analysis_name=flt_rawcnt_config['analysis_name'])

        # Create the directory where to save the counting
        suffix = 'counting'
        counting_dir_path, counting_gene_dirs = \
            utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,processing_hyb,
                            suffix,add_slash,flt_rawcnt_config['skip_tags_counting'],
                            flt_rawcnt_config['skip_genes_counting'],
                            analysis_name=flt_rawcnt_config['analysis_name'])


    
        for gene in hybridizations_infos[hybridization].keys():
            flist_img_to_filter=glob.glob(hyb_dir+processing_hyb+'_tmp/'+processing_hyb+'_'+gene+'_tmp/*.npy')
            # filtering
            logger.debug('Filtering without illumination correction %s',gene)

            futures_processes=client.map(counting.filtering_and_counting,flist_img_to_filter, \
                                    filtered_png_img_gene_dirs=filtered_png_img_gene_dirs, \
                                    filtered_img_gene_dirs=filtered_img_gene_dirs, \
                                    counting_gene_dirs=counting_gene_dirs, \
                                    plane_keep=flt_rawcnt_config['plane_keep'], min_distance=flt_rawcnt_config['min_distance'],\
                                    stringency=flt_rawcnt_config['stringency'],\
                                    skip_genes_counting=flt_rawcnt_config['skip_genes_counting'],skip_tags_counting=flt_rawcnt_config['skip_tags_counting'])

            client.gather(futures_processes)


        # ----------------- RAW COUNTING ONLY------------------------
        
        skip_genes_counting=flt_rawcnt_config['skip_genes_counting']
        skip_tags_counting=flt_rawcnt_config['skip_tags_counting']

        # Create the directory where to save the counting
        suffix = 'counting'
        counting_dir_path, counting_gene_dirs = \
            utils.create_subdirectory_tree(hyb_dir,hybridization,hybridizations_infos,processing_hyb,
                            suffix,add_slash,flt_rawcnt_config['skip_tags_counting'],
                            flt_rawcnt_config['skip_genes_counting'],
                            analysis_name=flt_rawcnt_config['analysis_name'])

        suffix = 'filtered_npy'
        gene_list = list(hybridizations_infos[hybridization].keys())
        analysis_name=flt_rawcnt_config['analysis_name']
        sufx_dir_path = hyb_dir+analysis_name+'_'+processing_hyb+'_'+suffix+add_slash
        
    
        for gene in hybridizations_infos[hybridization].keys():

            # Filtering image according to gene
            if gene not in skip_genes_counting or [tag for tag in skip_tags_counting if tag not in gene]:
                if analysis_name:
                    filtered_images_directory =  sufx_dir_path+analysis_name+'_'+processing_hyb+'_'+ gene+'_'+suffix+add_slash
                else:
                    filtered_images_directory =  sufx_dir_path +processing_hyb+'_'+ gene +'_'+suffix+add_slash
                
                flist_img_to_filter=glob.glob(hyb_dir+processing_hyb+'_tmp/'+processing_hyb+'_'+gene+'_tmp/*.npy')
                # filtering
                logger.debug('Filtering without illumination correction %s',gene)

                futures_processes=client.map(counting.counting_only,flist_img_to_filter, \
                                        counting_gene_dirs=counting_gene_dirs, \
                                        min_distance=flt_rawcnt_config['min_distance'],\
                                        stringency=flt_rawcnt_config['stringency'])

                client.gather(futures_processes)






    client.close()
    

if __name__ == '__main__':
    filtering_speed()