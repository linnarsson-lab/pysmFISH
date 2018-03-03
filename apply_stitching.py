import os
import sys
import numpy as np
import argparse
import h5py
import glob
import multiprocessing
from distributed import Client,LocalCluster
import logging
import re

# Own imports
from pysmFISH.stitching_package import stitching
from pysmFISH.stitching_package import tilejoining
from pysmFISH.stitching_package import hdf5preparation
from pysmFISH import utils



def apply_stitching():

    """
    Script to apply the registration to all the osmFISH channels. It will create
    a stitched image in an hdf5 file

    All the parameters are entered via argparse

    Parameters:
    -----------

    experiment_path: string
        Path to the folder with the hybridizations
    reference_files_path: string
        Path to the folder with the _reg_data.pkl files
    scheduler: string
        tcp address of the dask.distributed scheduler (ex. tcp://192.168.0.4:7003). 
        default = False. If False the process will run on the local computer using nCPUs-1


    """

    parser = argparse.ArgumentParser(description='Create the stitched images \
                                    after registration')

    parser.add_argument('-experiment_path', help='path to the folder with the hybridizations')
    parser.add_argument('-reference_files_path', help='path to the folder with the \
                        _reg_data.pkl files')
    parser.add_argument('-scheduler', default=False, help='dask scheduler address ex. tcp://192.168.0.4:7003')
    args = parser.parse_args()

    processing_experiment_directory = args.experiment_path
    stitched_reference_files_dir = args.reference_files_path
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


    # Determine the operating system running the code
    os_windows, add_slash = utils.determine_os()

    # Check training slash in the processing directory
    processing_experiment_directory=utils.check_trailing_slash(processing_experiment_directory,os_windows)

    stitched_reference_files_dir=utils.check_trailing_slash(stitched_reference_files_dir,os_windows)

    # Starting logger
    utils.init_file_logger(processing_experiment_directory)
    logger = logging.getLogger()


    # Collect the infos of the experiment and the processing
    # Parse the Experimental metadata file (serial)
    experiment_infos,image_properties, hybridizations_infos, \
    converted_positions, microscope_parameters =\
    utils.experimental_metadata_parser(processing_experiment_directory)

    # Parse the configuration file 
    flt_rawcnt_config = utils.filtering_raw_counting_config_parser(processing_experiment_directory)

    # Get the reference gene used
    reference_gene = flt_rawcnt_config['reference_gene']

    # Stitch the image in 2D or 3D (3D need more work/testing)
    nr_dim = flt_rawcnt_config['nr_dim']

    # Determine the hybridizations to process
    if isinstance(flt_rawcnt_config['hybs_to_stitch'],list):
        hybridizations_to_process = flt_rawcnt_config['hybs_to_stitch']
    else:
        if flt_rawcnt_config['hybs_to_stitch'] == 'All':
                hybridizations_to_process = list(hybridizations_infos.keys())
        
        else:
            raise ValueError('Error in the hybridizations to stitch')
    


    for hybridization in hybridizations_to_process:
        
        # Determine the genes to stitch in the processing hybridization
        genes_processing = list(hybridizations_infos[hybridization].keys())


        hyb_short = re.sub('Hybridization','hyb',hybridization)
        processing_hyb = experiment_infos['ExperimentName']+'_'+hyb_short
        hyb_dir = processing_experiment_directory+processing_hyb+add_slash

        # Create pointer of the hdf5 file that will store the stitched images
        # for the current hybridization

        tile_file_base_name = flt_rawcnt_config['analysis_name']+'_'+ processing_hyb
        stitching_file_name = tile_file_base_name + '.reg.sf.hdf5'

        data_name = (tile_file_base_name
                            + '_' + reference_gene
                            + '_stitching_data_reg')

        stitching_file= h5py.File(hyb_dir+stitching_file_name,'w',libver='latest')  # replace with 'a' as soon as you fix the error

        # Determine the tiles organization
        joining, tiles, nr_pixels, z_count, micData = stitching.get_place_tile_input_apply_npy(hyb_dir,stitched_reference_files_dir,data_name,image_properties,nr_dim)

        for gene in genes_processing:
        
            # Create the hdf5 file structure
            stitched_group, linear_blending, blend =  hdf5preparation.create_structures_hdf5_stitched_ref_gene_file_npy(stitching_file, joining, nr_pixels,
                                            gene, blend = 'non linear')

            # Fill the hdf5 containing the stitched image with empty data and
            # create the blending mask
            stitched_group['final_image'][:]= np.zeros(joining['final_image_shape'],dtype=np.uint16)
            if blend is not None:
                # make mask
                stitched_group['blending_mask'][:] = np.zeros(joining['final_image_shape'][-2:],dtype=np.uint16)
                tilejoining.make_mask(joining, nr_pixels, stitched_group['blending_mask'])

            filtered_img_gene_dirs_path = hyb_dir+flt_rawcnt_config['analysis_name']+'_'+processing_hyb +'_filtered_npy'+add_slash
            filtered_img_gene_dirs = glob.glob(filtered_img_gene_dirs_path+'*')

            # Create the subdirectory used to save the blended tiles
            suffix = 'blended_tiles'
            blended_tiles_directory = utils.create_single_directory(hyb_dir,gene, hybridization,processing_hyb,suffix,add_slash,
                                            analysis_name=flt_rawcnt_config['analysis_name'])

            # Get the directory with the filtered npy images of the reference_gene to use for stitching
            stitching_files_dir = [npy_dir for npy_dir in filtered_img_gene_dirs if gene in npy_dir][0]
            stitching_files_dir= stitching_files_dir+add_slash

            # Create the tmp directory where to save the masks
            suffix = 'masks'
            masked_tiles_directory = utils.create_single_directory(hyb_dir,gene,hybridization,processing_hyb,suffix,add_slash,
                                            analysis_name=flt_rawcnt_config['analysis_name'])

            # Create and save the mask files
            for corn_value,corner_coords in joining['corner_list']:
                if not(np.isnan(corner_coords[0])):
                    cur_mask = stitched_group['blending_mask'][int(corner_coords[0]):int(corner_coords[0]) + int(nr_pixels),
                                        int(corner_coords[1]):int(corner_coords[1]) + int(nr_pixels)]

                    fname = masked_tiles_directory + flt_rawcnt_config['analysis_name'] +'_'+processing_hyb+'_'+gene+'_masks_joining_pos_'+str(corn_value)
                    np.save(fname,cur_mask)


            # Blend all the tiles and save them in a directory
            futures_processes = client.map(tilejoining.generate_blended_tile_npy,joining['corner_list'],
                                        stitching_files_dir = stitching_files_dir,
                                        blended_tiles_directory = blended_tiles_directory,
                                        masked_tiles_directory = masked_tiles_directory,
                                        analysis_name = flt_rawcnt_config['analysis_name'],
                                        processing_hyb = processing_hyb,reference_gene = gene,
                                        micData = micData,tiles = tiles,nr_pixels=nr_pixels,
                                        linear_blending=linear_blending)



            _ = client.gather(futures_processes)


            # Write the stitched image
            tilejoining.make_final_image_npy(joining, stitching_file, blended_tiles_directory, tiles,gene, nr_pixels)
            stitching_file.flush()

            # Remove directories with blended tiles and masks
            shutil.rmtree(blended_tiles_directory)
            shutil.rmtree(masked_tiles_directory)


        stitching_file.close()

    
    client.close()


if __name__ == "__main__":
    apply_stitching()