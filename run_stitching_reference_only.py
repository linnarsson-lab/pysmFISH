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

from pysmFISH import utils,counting,dots_calling,io, hdf5_utils

from pysmFISH.stitching_package import stitching
from pysmFISH.stitching_package import tilejoining
from pysmFISH.stitching_package import hdf5preparation
from pysmFISH.stitching_package import pairwisesingle



def run_stitching_reference_only():
    
    """
    This script will stitch the filtered data of the hybridizations
    in experiment directory. The inputs parameters are passed using arparse

    Parameters:
    -----------
    
    scheduler: string
        tcp address of the dask.distributed scheduler (ex. tcp://192.168.0.4:7003). 
        default = False. If False the process will run on the local computer using nCPUs-1

    path: string
        Path to the experiment directory

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
    # utils.init_file_logger(processing_directory)
    # logger = logging.getLogger()

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


        # ----------------- STITCHING ------------------------

        # Determine the directory of the filtered images
        
        suffix = 'filtered_npy'
        analysis_name=flt_rawcnt_config['analysis_name']
        sufx_dir_path = hyb_dir+analysis_name+'_'+processing_hyb+'_'+suffix+add_slash
        
        # Reference gene for stitching
        reference_gene = flt_rawcnt_config['reference_gene']

        filtered_gene_dir = sufx_dir_path+analysis_name+'_'+processing_hyb+'_'+reference_gene+'_'+suffix+add_slash

        # Load the stitching parameters from the .yaml file

        # Stitch the image in 2D or 3D (3D need more work/testing)
        nr_dim = flt_rawcnt_config['nr_dim']

        # Estimated overlapping between images according to the Nikon software
        est_overlap = np.float_(image_properties['Overlapping_percentage'])
        

        # Number of peaks to use for the alignment
        nr_peaks = flt_rawcnt_config['nr_peaks']

        # Determine if the coords need to be flipped

        y_flip = flt_rawcnt_config['y_flip']

        # Method to use for blending
        # can be 'linear' or 'non linear'
        # The methods that performs the best is the 'non linear'

        blend = flt_rawcnt_config['blend']


        pixel_size = image_properties['PixelSize']

        # Get the list of the filtered files of the reference gene
        filtered_files_list = glob.glob(filtered_gene_dir+'*.npy')

        # Create pointer of the hdf5 file that will store the stitched reference image
        # for the current hybridization
        # Writing
        tile_file_base_name = flt_rawcnt_config['analysis_name']+'_'+ processing_hyb
        data_name   = (tile_file_base_name
                        + '_' + reference_gene
                        + '_stitching_data')

        stitching_file_name = tile_file_base_name + '.sf.hdf5'
        stitching_file= h5py.File(hyb_dir+stitching_file_name,'w',libver='latest')  # replace with 'a' as soon as you fix the error


        # Determine the tiles organization
        tiles, contig_tuples, nr_pixels, z_count, micData = stitching.get_pairwise_input_npy(image_properties,converted_positions, hybridization,
                                est_overlap, y_flip = False, nr_dim = 2)


        # Align the tiles 
        futures_processes=client.map(pairwisesingle.align_single_pair_npy,contig_tuples,
                                    filtered_files_list=filtered_files_list,micData=micData, 
                                nr_peaks=nr_peaks)

        # Gather the futures
        data = client.gather(futures_processes)


        # In this case the order of the returned contingency tuples is with
        # the order of the input contig_tuples

        # P_all = [el for data_single in data for el in data_single[0]]
        P_all =[data_single[0] for data_single in data ]
        P_all = np.array(P_all)
        P_all = P_all.flat[:]
        covs_all = [data_single[1] for data_single in data]
        alignment = {'P': P_all,
                    'covs': covs_all}


        # Calculates a shift in global coordinates for each tile (global
        # alignment) and then applies these shifts to the  corner coordinates
        # of each tile and returns and saves these shifted corner coordinates.
        joining = stitching.get_place_tile_input(hyb_dir, tiles, contig_tuples,
                                                    micData, nr_pixels, z_count,
                                                    alignment, data_name,
                                                    nr_dim=nr_dim)

        # Create the hdf5 file structure
        stitched_group, linear_blending, blend =  hdf5preparation.create_structures_hdf5_stitched_ref_gene_file_npy(stitching_file, joining, nr_pixels,
                                        reference_gene, blend = 'non linear')

        # Fill the hdf5 containing the stitched image with empty data and
        # create the blending mask
        stitched_group['final_image'][:]= np.zeros(joining['final_image_shape'],dtype=np.float64)
        if blend is not None:
            # make mask
            stitched_group['blending_mask'][:] = np.zeros(joining['final_image_shape'][-2:],dtype=np.float64)
            tilejoining.make_mask(joining, nr_pixels, stitched_group['blending_mask'])

            
        # Create the subdirectory used to save the blended tiles
        suffix = 'blended_tiles'
        blended_tiles_directory = utils.create_single_directory(hyb_dir,reference_gene, hybridization,processing_hyb,suffix,add_slash,
                                        analysis_name=flt_rawcnt_config['analysis_name'])


        # Create the tmp directory where to save the masks
        suffix = 'masks'
        masked_tiles_directory = utils.create_single_directory(hyb_dir,reference_gene, hybridization,processing_hyb,suffix,add_slash,
                                        analysis_name=flt_rawcnt_config['analysis_name'])

        # Create and save the mask files
        for corn_value,corner_coords in joining['corner_list']:
            if not(np.isnan(corner_coords[0])):
                cur_mask = stitched_group['blending_mask'][int(corner_coords[0]):int(corner_coords[0]) + int(nr_pixels),
                                    int(corner_coords[1]):int(corner_coords[1]) + int(nr_pixels)]

                fname = masked_tiles_directory + flt_rawcnt_config['analysis_name'] +'_'+processing_hyb+'_'+reference_gene+'_masks_joining_pos_'+str(corn_value)
                np.save(fname,cur_mask)



        # Blend all the tiles and save them in a directory
        futures_processes = client.map(tilejoining.generate_blended_tile_npy,joining['corner_list'],
                                    stitching_files_dir = filtered_gene_dir,
                                    blended_tiles_directory = blended_tiles_directory,
                                    masked_tiles_directory = masked_tiles_directory,
                                    analysis_name = flt_rawcnt_config['analysis_name'],
                                    processing_hyb = processing_hyb,reference_gene = reference_gene,
                                    micData = micData,tiles = tiles,nr_pixels=nr_pixels,
                                    linear_blending=linear_blending)



        _ = client.gather(futures_processes)


        # Write the stitched image
        tilejoining.make_final_image_npy(joining, stitching_file, blended_tiles_directory, tiles,reference_gene, nr_pixels)

        # close the hdf5 file
        stitching_file.close()


        # Delete the directories with blended tiles and masks
        shutil.rmtree(blended_tiles_directory)
        shutil.rmtree(masked_tiles_directory)

   
    client.close()
    

if __name__ == '__main__':
    run_stitching_reference_only()