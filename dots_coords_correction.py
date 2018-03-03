import argparse
from distributed import Client,LocalCluster
import multiprocessing
import pickle

from pysmFISH import dots_coords_calculations
from pysmFISH import utils



def dots_coords_correction():
    """
    This script is used to collect all the raw countings from the different
    hybridization, correct the coords according to the registration of the 
    reference gene and remove the dots that overlap in the overlapping
    regions between the images. Save the aggregate coords and also the coords after dots processing

    Input via argparse

    Parameters:
    -----------

    path: string. 
        Exact path to the experiment folder
    pxl: int 
        Radius of pixel used to create the neighbourhood (nhood) used to define 
        when two dots are the same
    
    """

    # Inputs of the function
    parser = argparse.ArgumentParser(description='Dots coords consolidation \
                                    and correction')

    parser.add_argument('-path', help='path to the experiment folder')
    parser.add_argument('-pixel_radius', help='adius of pixel used to create the nhood \
                            that is used to define that two pixels are the same', 
                            type=int)
    parser.add_argument('-scheduler', default=False, help='dask scheduler address ex. tcp://192.168.0.4:7003')
    
    args = parser.parse_args()

    # retrieve the parameters
    processing_experiment_directory = args.path
    pxl = args.pixel_radius

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

    # Check training slash in the experiment directory
    processing_experiment_directory=utils.check_trailing_slash(processing_experiment_directory,os_windows)

    stitched_reference_files_dir = processing_experiment_directory + 'stitched_reference_files'

    # Check training slash in the stitched reference directory
    stitched_reference_files_dir=utils.check_trailing_slash(stitched_reference_files_dir,os_windows)

    # Collect the infos of the experiment and the processing
    # Parse the Experimental metadata file (serial)
    experiment_infos,image_properties, hybridizations_infos, \
    converted_positions, microscope_parameters =\
    utils.experimental_metadata_parser(processing_experiment_directory)

    # Parse the configuration file 
    flt_rawcnt_config = utils.filtering_raw_counting_config_parser(processing_experiment_directory)


    # get the reference gene
    reference_gene = flt_rawcnt_config['reference_gene']

    # get the overlapping percentage and image_size
    overlapping_percentage = image_properties['Overlapping_percentage']

    # Consider a square image
    image_size = image_properties['HybImageSize']['columns']

    # Combine all counts
    all_raw_counts = dots_coords_calculations.combine_raw_counting_results(flt_rawcnt_config,
                                    hybridizations_infos,experiment_infos,
                                    processing_experiment_directory,stitched_reference_files_dir,
                                    reference_gene,add_slash)

    # Create a dictionary with only the selected peaks coords after alignment
    aligned_peaks_dict = all_raw_counts['selected_peaks_coords_aligned']

    # Create list of tuples to process each hybridization/gene on a different core
    combinations = dots_coords_calculations.processing_combinations(list(hybridizations_infos.keys()),aligned_peaks_dict)

    # Add corresponding registration_data and the corresponding coords files to the
    # tuple is order to recduce the size of the info transferred in the newtwork
    added_combinations =list()
    for idx,combination in enumerate(combinations):
        hybridization = combination[0]
        gene = combination[1]
        reg_data_combination = all_raw_counts['registration_data'][hybridization]
        aligned_peaks_dict_gene = all_raw_counts['selected_peaks_coords_aligned'][hybridization][gene]
        combination_dict = {
                'hybridization':hybridization,
                'gene':gene,
                'reg_data_combination':reg_data_combination,
                'aligned_peaks_dict_gene': aligned_peaks_dict_gene
        }
        added_combinations.append(combination_dict)

        # Process each gene in parallel
        futures_processes = client.map(dots_coords_calculations.function_to_run_dots_removal_parallel,added_combinations,
                            overlapping_percentage = overlapping_percentage,
                            image_size = image_size,pxl = pxl)

        cleaned_dots_list = client.gather(futures_processes)

    # Convert the list of dictionaries in one single dictionary
    # The saved dictionary cotains all the dots, the reference to the tile pos
    # has been removed during the overlapping dots removal step

    all_countings = dict()
    all_countings['all_coords_cleaned'] = dict()
    all_countings['all_coords'] = dict()
    all_countings['removed_coords'] = dict()

    for el in cleaned_dots_list:
        hybridization = list(el.keys())[0]
        gene = list(el[hybridization].keys())[0]
        
        renamed_gene = gene + '_' + hybridization
        
        all_countings['all_coords_cleaned'][renamed_gene] = el[hybridization][gene]['all_coords_cleaned']
        all_countings['all_coords'][renamed_gene] = el[hybridization][gene]['all_coords']
        all_countings['removed_coords'][renamed_gene] = el[hybridization][gene]['removed_coords']

    # Save all the data
    counting_data_name = processing_experiment_directory +experiment_infos['ExperimentName']+'_all_cleaned_raw_counting_data.pkl'
    pickle.dump(all_countings,open(counting_data_name,'wb'))

    client.close()


if __name__ == "__main__":
    dots_coords_correction()