
import h5py
import numpy as np
from skimage import filters,io,img_as_float,exposure,morphology,segmentation,measure,feature,color
from scipy import ndimage as nd
import itertools
import pickle
import multiprocessing

from pysmFISH import utils
from pysmFISH import object_based_segmentation

def staining_segmentation():

    """
    This script will segment the selected staining and output the identified 
    objects.

    All the parameters are entered via argparse.

    Parameters:
    -----------

    scheduler: string
        tcp address of the dask.distributed scheduler (ex. tcp://192.168.0.4:7003). 
        default = False. If False the process will run on the local computer using nCPUs-1
    path: string
        Path to the processing directory
    processing_file: string
        Path to the hdf5 file with the staning to process
    segmentation_staining: string
        Staining to be segmented

    """

    # Inputs of the function
    parser = argparse.ArgumentParser(description='Segmentation script')
    parser.add_argument('-scheduler', default=False, help='dask scheduler address ex. tcp://192.168.0.4:7003')
    parser.add_argument('-path', help='processing directory')
    parser.add_argument('-processing_file', help='path to the file with the \
                        staning to process')
    parser.add_argument('-segmentation_staining', help='staining to be \
                        segmented')
    
    

    args = parser.parse_args()
    
    # Directory to process
    processing_directory = args.path

    # File to process
    processing_file = args.processing_file

    # staining to segment
    segmentation_staining = args.segmentation_staining

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
    processing_directory=utils.check_trailing_slash(processing_directory,os_windows)


    segmentation_parameters = utils.general_yaml_parser(processing_directory+'Staining_segmentation.config.yaml')

    # Chunking parameters
    chunk_size = segmentation_parameters[segmentation_staining]['image_chunking_parameters']['chunk_size']
    percent_padding = segmentation_parameters[segmentation_staining]['image_chunking_parameters']['percent_padding']

    # Segmentation parameters
    trimming = segmentation_parameters[segmentation_staining]['segmentation_parameters']['trimming']
    min_object_size = segmentation_parameters[segmentation_staining]['segmentation_parameters']['min_object_size']
    disk_radium_rank_filer = segmentation_parameters[segmentation_staining]['segmentation_parameters']['disk_radium_rank_filer']
    min_distance = segmentation_parameters[segmentation_staining]['segmentation_parameters']['min_distance']   
    threshold_rel = segmentation_parameters[segmentation_staining]['segmentation_parameters']['threshold_rel']    

    # Load the image (will be modified after the change to hdf5 input)
    img = io.imread(processing_file)

    # Image chunking
    nr_chunks,nc_chunks,Coords_Chunks_list, Coords_Padded_Chunks_list,r_coords_tl_all_padded,\
                c_coords_tl_all_padded,r_coords_br_all_padded,c_coords_br_all_padded = \
                object_based_segmentation.image_chunking(img,chunk_size,percent_padding)
            
            
    # Create the chunks idx
    Chunks_idxs_linear=np.arange(len(Coords_Padded_Chunks_list),dtype='int32')

    # Distribute the chunks idx and distridute them in an array according to the position
    # in the chunked image
    Chunks_idxs=Chunks_idxs_linear.reshape(nr_chunks,nc_chunks)  

    # Flatten the array for make it easier the creation of the coords combination
    Chunks_idxs_rows=np.ravel(Chunks_idxs)
    Chunks_idxs_cols=np.ravel(Chunks_idxs,order='F')

    # Calculate coords of the overlapping chunks
    Overlapping_chunks_coords=list()
    counter=0
    left_pos=Chunks_idxs_rows[0]
    for el in Chunks_idxs_rows[1:]:
        if counter < nc_chunks-1:
            Coords_left=Coords_Padded_Chunks_list[left_pos]
            Coords_right=Coords_Padded_Chunks_list[el]
            row_tl=Coords_left[0]
            row_br=Coords_left[1]
            col_tl=Coords_right[2]
            col_br=Coords_left[3]
            Overlapping_chunks_coords.append((row_tl,row_br,col_tl,col_br))
            left_pos=el
            counter+=1
        else:
            left_pos=el
            counter=0
            
    counter=0
    top_pos=Chunks_idxs_cols[0]
    for el in Chunks_idxs_cols[1:]:
        if counter < nr_chunks-1:
            Coords_top=Coords_Padded_Chunks_list[top_pos]
            Coords_bottom=Coords_Padded_Chunks_list[el]
            
            row_tl=Coords_bottom[0]
            row_br=Coords_top[1]
            col_tl=Coords_top[2]
            col_br=Coords_top[3]
            Overlapping_chunks_coords.append((row_tl,row_br,col_tl,col_br))
        
            counter+=1
            top_pos=el
        else:
            top_pos=el
            counter=0

    # Now i use this approach for testing. If the image gets to big to fit in RAM
    # then save the files and load them separately in each node
    chunked_image_seq = list()
    for coords in Coords_Padded_Chunks_list:
        chunked_image_seq.append(img[coords[0]:coords[1],coords[2]:coords[3]])

    # Run the segmentation
    futures_processes = client.map(object_based_segmentation.polyT_segmentation,chunked_image_seq,
                                min_object_size=min_object_size,
                                min_distance=min_distance,
                                disk_radium_rank_filer=disk_radium_rank_filer,
                                threshold_rel=threshold_rel,
                                trimming=trimming)

    Objects_list = client.gather(futures_processes)

    # Recalculate labels and coords

    processed_images_data = dict()

    max_starting_label = 0
    total_data_dict = dict()

    for idx, objs_chunk in enumerate(Objects_list):
        for label ,cvalues in objs_chunk.items():
                new_label=max_starting_label+1
                coords = Coords_Padded_Chunks_list[idx][0::2]
                total_data_dict[new_label] = cvalues+coords
                max_starting_label = new_label


    # Calculate all the intersecting objects
    futures_processes = client.map(object_based_segmentation.OverlappingCouples,Overlapping_chunks_coords,
                                TotalDataDict = total_data_dict)
        

    All_intersecting = client.gather(futures_processes)


    # Put together the couple with the same label for multiple intersection
    # for the labels of objects where there is intersection between multiple regions
    # Then scatter all of them and calculate intersection

    # Combine the results from the parallel processing
    flatten_couple = [el for grp in All_intersecting for el in grp] 
    # Remove duplicates
    flatten_couple=list(set(flatten_couple))

    # Create a list of the labels (removing the repeats)
    singles=list()
    [singles.append(x) for cpl in flatten_couple for x in cpl]
    singles=list(set(singles))

    # Identify the couples containing singles
    Combined_all_singles=list()
    for item in singles:
        Combined_single=list()
        for couple in flatten_couple:
            if item in couple:
                Combined_single.append(couple)
        Combined_all_singles.append(Combined_single)

    if Combined_all_singles:
        # Combine all the intersecting labeles
        start=Combined_all_singles[0]
        ComparisonList=Combined_all_singles[1:].copy()
        #merged=start.copy()
        merged=list()
        SavedCombinations=list()
        tmp_list=ComparisonList.copy()
        KeepGoing=True
        Loop=0
        while KeepGoing:
            Loop+=1


            for idx,el in enumerate(ComparisonList):

                if set(start).intersection(set(el)):
                    #merged=list(set(merged)|set(el))
                    [merged.append(x) for x in el]
                    tmp_list = [e for e in tmp_list if e != el]


            intersection=list(set.intersection(set(merged),set(start)))  
            if intersection:
                merged=list(set.union(set(merged),set(start)))
                #merged=list(set(merged))
                start=merged.copy()
                merged=list()
                ComparisonList=tmp_list.copy()
                #tmp_list.append(merged)
            else:
                SavedCombinations.append(start)
                start=tmp_list[0]
                tmp_list=tmp_list[1:]
                ComparisonList=tmp_list.copy()


            if len(tmp_list)<1:
                [SavedCombinations.append(x) for x in tmp_list]
                KeepGoing =False
        
        # Remove all the duplicated labeled that intersect
        # in this case the labeled are merged. It will be nice to run an extra
        # segmentation on the merged objects
        # If it is too slow this step can be parallelised

        SavedLab_list=list()
        CleanedDict=total_data_dict.copy()
        for couple in SavedCombinations:
            SaveLab, RemoveLabs,NewCoords=object_based_segmentation.IntersectionCouples(couple,total_data_dict)
            SavedLab_list.append(SaveLab)
            for lab in RemoveLabs:
                del CleanedDict[lab]
            CleanedDict[SaveLab]=NewCoords
    else:
        CleanedDict=total_data_dict 


    # Calculate all objects properties
    all_objects_list = [(key,coords) for key,coords in CleanedDict.items()]

    futures_processes = client.map(object_based_segmentation.obj_properties_calculator,all_objects_list)

    all_objects_properties_list = client.gather(futures_processes)


    # convert the list to a dictionary
    all_objects_properties_dict = { k: v for d in all_objects_properties_list for k, v in d.items() }

    # Save all the objects
    segmented_objs_fname = processing_directory + 'segmented_' + segmentation_staining + '_all_objs_properties.pkl'
    pickle.dump(all_objects_properties_dict,open(segmented_objs_fname,'wb'))


if __name__ == "__main__":
    staining_segmentation()