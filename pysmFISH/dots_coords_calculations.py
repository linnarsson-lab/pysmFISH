import numpy as np
import pickle
import glob
import re

def register_dots_coords(reg_data,hybridization,gene,all_raw_counts):
    
    """
    Function to register the coords of the dots according to the new coords
    of the corner of the images calculated after registration of the stitched 
    images.
    
    Parameters:
    -----------

    reg_data: dict 
        Dictionary containing the coords and the joining
        data calculated after the registration.
    hybridization: str 
        Name of the hybridization processed (Ex. Hybridization2)
    gene: str 
        Processed gene name (Ex. Gfap)
    all_raw_counts: dict 
        Dictionary with all the raw counting from all the hybridizations
    
    Returns:
    ---------

    all_raw_counts: dict 
        Dictionary with all the raw counting from all the hybridizations. 
        Added a subgroup containing the aligned coords.
    """
    
    
    tile_set = reg_data['micData'].tile_set.data
    tile_set = tile_set.ravel()
    for pos in all_raw_counts['selected_peaks_coords_not_aligned'][hybridization][gene].keys():  
        idx = np.where(tile_set==np.int(pos))[0][0]
        corner_coords = reg_data['joining']['corner_list'][idx][1]
        old_coords = all_raw_counts['selected_peaks_coords_not_aligned'][hybridization][gene][pos]
        if not np.all(old_coords==0):
            corrected_coords = old_coords + corner_coords
            all_raw_counts['selected_peaks_coords_aligned'][hybridization][gene][pos] = corrected_coords
    return all_raw_counts


def processing_combinations(hybridizations,aligned_peaks_dict):
    
    """
    Function used to create a list of tuples containing the pair hybridization
    and gene (Ex. (Hybridization1, Gfap)) that will be used to process the dots
    removal in parallel.
    
    Parameters:
    -----------

    hybridizations: list 
        List of the hybridization to process 
        (ex. [Hybridization1, Hybridization2]).
    aligned_peaks_dict: dict 
        Dictionary containing only the coords of the selected peaks after 
        registration (still with pos information).
                                
                                
    Returns:
    ---------
    combinations: list 
        List containing the pairs hybridization and gene 
        (Ex. (Hybridization1, Gfap))
    
    """  
    
    combinations_tmp=list()
    if isinstance(hybridizations,list):
        for hybridization in hybridizations:
            combinations_tmp.append([(hybridization,gene) for gene in aligned_peaks_dict[hybridization].keys()])
        combinations=[el for grp in combinations_tmp for el in grp]
    else:
        hybridization=hybridizations
        combinations=[(hybridization,gene) for gene in aligned_peaks_dict[hybridization].keys()]
    return combinations


def overlapping_images_grid(micData):
    """
    Function used to generate all possible overlapping images (neighborhood).
    
    Parameters:
    -----------

    micData: dict 
        Dictionary extracted from the reg_data and containing the infomation 
        regarding tiles organization
    
    Returns:
    ---------

    all_nhood: list 
        List with all the overlapping tiles combinations
    
    """
    
    tile_set=micData.tile_set.data
    Img_r,Img_c=tile_set.shape
    all_nhood=list()

    for r in np.arange(Img_r):
        for c in np.arange(Img_c):
            # Skip the position with the center tile ==-1 in the ImgSet
            if tile_set[r,c]!=-1:
                # I know the order of the neighboouring objects
                coordinates=[(r+i, c+j) for j in np.arange(-1,2) for i in np.arange(-1,2)
                             if ((0<=r+i<=Img_r-1) and (0<=c+j<=Img_c-1))]
                # Remove the pairs where -1 is in the ImgSet
                coordinates=[i for i in coordinates if tile_set[i]!=-1 ]
                if coordinates:
                    all_nhood.append(tuple(coordinates)) 
    return tuple(all_nhood)


def hood_creation(coords,pxl):
    
    """
    Function that creates the neighborood of coords that are considered 
    same RNA molecule. It is the region used to eliminate overlapping dots.
    
    Parameters:
    -----------

    coords: np.array 
        Array of the [r,c] of the dots.
    pxl: int
        Radius of pixel used to create the nhood.
    
    Return:
    ---------

    hood: set 
        Set with the coords of the neighborhood of the selected dot.
    
    """
    
    rg=np.arange(-(pxl),pxl+1)
    hood=tuple([(coords[0]+i,coords[1]+j) for i in rg for j in rg])
    return set(hood)


def dots_removing(overlapping_list,pxl):
    """
    Function used to identify matching dots that are present in overlapping 
    imaging areas.
    
    Parameters:
    -----------

    overlapping_list: list 
        List with the coords of the dots located in the overlapping region.                    
    pxl: int 
        Radius of pixel used to create the nhood.


    Returns:
    ---------

    removed_dots: set 
        Set with the coords of the removed dots.
    single_dots: set 
        Set with the coords of dots in the overlpping region that do not 
        overlap with any other dot.
    
    """
    
    removed_dots_coords=list()
    single_dots=set(overlapping_list.copy())
    remove_dot_step={}
    for coords in overlapping_list:
        if not {coords}.intersection(remove_dot_step):
            hood=hood_creation(coords,pxl)
            # Remove pxl to keep from hood
            hood=hood.difference({coords})
            single_dots=single_dots.difference(hood)
            remove_dot_step=set(overlapping_list).difference(set(single_dots))
    removed_dots=set(overlapping_list).difference(set(single_dots))
    return removed_dots,single_dots


def catch_duplicated_dots(combination, image_list,
                          overlapping_percentage,image_size,pxl):
    
    # dots_removing
    
    """
    Function used to identify dots that need to be removed or kept in the
    overlapping region between two images. The overlapping used is calculated
    according to the stitching information.
    
    Parameters:
    -----------

    combination: dict 
        Dictionary containing hybridization, gene, registation data of the 
        current hybridization and the registerd coords of the selected peaks 
        for the current gene.
    image_list: tuple 
        Tuple with the index of the the images with ooverlap
    overlapping_percentage: float 
        Percentage of overlap extracted from the Experimental_metadata.yaml.
    image_size: int 
        Size of the image (consider a square image).     
    pxl: int 
        Radius of pixel used to create the nhood.
    
    Returns:
    -----------

    overlapping_result_dict: dict 
        Dictionary with the coords of the dots that need to be removed or will 
        be kept for the selected pair of images.
    
    """
     
    # Extract the data
    micData = combination['reg_data_combination']['micData']
    joining = combination['reg_data_combination']['joining']
    aligned_peaks_dict_gene = combination['aligned_peaks_dict_gene']
    gene = combination['gene']
    hybridization = combination['hybridization']
    
    
    # Determine the image set from the stitched data
    img_set = micData.tile_set.data
    img_set_rav=img_set.ravel()

    # Create the overlapping results dict
    overlapping_result_dict=dict()
    overlapping_result_dict['row_dir']={}
    overlapping_result_dict['col_dir']={}

    overlapping_pixels=overlapping_percentage*image_size
    
    # Sort the list by row
    image_list=list(image_list)
    image_list.sort(key=lambda tup: tup[0]) 

    # work on rows
    rows=[el[0] for el in image_list]
    rows=set(rows)

    cols=[el[1] for el in image_list]
    cols=set(cols)

    overlapping=list()
    non_overlapping=list()
    counter=0

    # Process the images in rows
    for r in rows:
        row_grp=[ el for el in image_list if el[0] == r ]

        # Calculate the overlapping
        for p in np.arange(len(row_grp)-1):
            overlapping=list()
            non_overlapping=list()

            r,c=row_grp[p]
            pos_l=img_set[r,c]
            idx=np.where(img_set_rav==np.int(pos_l))[0][0]
            corner_left=joining['corner_list'][idx]
            corner_coords_left=corner_left[1]

            r,c=row_grp[p+1]
            pos_r=img_set[r,c]
            idx=np.where(img_set_rav==np.int(pos_r))[0][0]
            corner_right=joining['corner_list'][idx]
            corner_coords_right=corner_right[1]

            positions=[pos_l,pos_r]

            # Calculation expected coords in cols
            expected_corner_c=corner_coords_left[1]+image_size-overlapping_pixels
            c_error=np.abs(corner_coords_right[1]-expected_corner_c)
            r_error=np.abs(corner_coords_left[0]-corner_coords_right[0])

            # Remember I know the order of image acquisition
            # Determine the overlapping bounding-box
            r_TL=np.maximum(corner_coords_left[0],corner_coords_right[0])
            c_TL=corner_coords_right[1]
            TL=(r_TL,c_TL)

            r_BR=np.minimum(corner_coords_left[0]+image_size,corner_coords_right[0]+image_size)
            c_BR=corner_coords_left[1]+image_size
            BR=(r_BR,c_BR)


            # identify the dots in the overlapping region
            for pos in positions:
                if pos in aligned_peaks_dict_gene.keys():
                    dots=aligned_peaks_dict_gene[pos]

                    for dot in dots:
                            if (dot>=TL).all() and (dot<=BR).all():
                                overlapping.append(dot)
                            else:
                                non_overlapping.append(dot)

            # Fix the counting by removing the dots
            overlapping_list=[tuple(el) for el in overlapping]
            removed_dots,single_dots=dots_removing(overlapping_list,pxl)
            overlapping_result_dict['row_dir'][counter]={'tl':TL,'br':BR,'removed_dots':removed_dots,
                                                          'single_dots':single_dots}

            counter+=1


    counter=0 

    # Process the images in columns
    for c in cols:
        col_grp=[ el for el in image_list if el[1] == c ]

        # Calculate the overlapping
        for p in np.arange(len(col_grp)-1):
            overlapping=list()
            non_overlapping=list()

            r,c=col_grp[p]
            pos_t=img_set[r,c]

            idx=np.where(img_set_rav==np.int(pos_t))[0][0]
            corner_top=joining['corner_list'][idx]
            corner_coords_top=corner_top[1]
            r,c=col_grp[p+1]
            pos_b=img_set[r,c]

            idx=np.where(img_set_rav==np.int(pos_b))[0][0]
            corner_bottom=joining['corner_list'][idx]
            corner_coords_bottom=corner_bottom[1]

            positions=[pos_t,pos_b]

            # Calculation expecated coords in cols
            expected_corner_r=corner_coords_top[0]+image_size-overlapping_pixels
            r_error=np.abs(corner_coords_bottom[0]-expected_corner_r)
            c_error=np.abs(corner_coords_top[1]-corner_coords_bottom[1])

            # Remember I know the order of image acquisition
            # Determine the overlapping bounding-box
            r_TL=corner_coords_bottom[0]
            c_TL=np.maximum(corner_coords_bottom[1],corner_coords_top[1])
            TL=(r_TL,c_TL)


            r_BR=corner_coords_top[0]+image_size
            c_BR=np.minimum(corner_coords_bottom[1]+image_size,corner_coords_top[1]+image_size)
            BR=(r_BR,c_BR)


            # identify the dots in the overlapping region
            for pos in positions:
                if pos in aligned_peaks_dict_gene.keys():
                    dots=aligned_peaks_dict_gene[pos]

                    for dot in dots:
                            if (dot>=TL).all() and (dot<=BR).all():
                                overlapping.append(dot)
                            else:
                                non_overlapping.append(dot)

                # Fix the counting by removing the dots
                overlapping_list=[tuple(el) for el in overlapping]
                removed_dots,single_dots=dots_removing(overlapping_list,pxl)
                overlapping_result_dict['col_dir'][counter]={'tl':TL,'br':BR,'removed_dots':removed_dots,
                                                              'single_dots':single_dots}

                counter+=1

    
    return overlapping_result_dict


def all_dots_removal(combination,all_removed):

    """
    Function to remove the overlapping dots.
    
    Parameters:
    -----------

    combination: dict  
        Dictionary containing hybridization, gene, registation data of the 
        current hybridization and the registerd coords of the selected peaks 
        for the current gene.
    all_removed: dict 
        Dictionary with the coords of the dots that need to be removed or will 
        be kept for all the images

    Returns:
    -----------

    all_coords_cleaned: set 
        Set containing the coords of the dots after removal of the overlapping 
        ones. No image pos info is kept.
    removed_coords: set 
        Set containing the coords of the dots removed. No image pos info is kept.
    all_coords: set 
        Set containing the coords of all the aligned dots before removal
        of the overlapping ones. No image pos info is kept.
    
    
    """
    aligned_peaks_dict_gene = combination['aligned_peaks_dict_gene']
    gene = combination['gene']
    hybridization = combination['hybridization']
    
    # Create the set of the coords that will be removed
    directions_list=['row_dir','col_dir']
    removed_coords=set()
    for direction in directions_list:
        for idx in all_removed[hybridization][gene].keys():
                for pos in all_removed[hybridization][gene][idx][direction].keys():
                    if all_removed[hybridization][gene][idx][direction][pos]['removed_dots']:
                        removed_coords=removed_coords.union(all_removed[hybridization][gene][idx][direction][pos]['removed_dots'])

    # Create a set with the coords of all the dots
    all_coords=list()
    for pos in aligned_peaks_dict_gene.keys():
        all_coords.append([tuple(coords) for coords in aligned_peaks_dict_gene[pos]])

    all_coords=[el for coords in all_coords for el in coords]
    all_coords=set(all_coords)


    # Remove the coords from the dataset
    all_coords_cleaned=all_coords.difference(removed_coords)
    
    return all_coords_cleaned,removed_coords,all_coords

def function_to_run_dots_removal_parallel(combination,overlapping_percentage,image_size,pxl):
    
    """
    Function to run the overlapping dots removal in parallel
    
    Parameters:
    -----------

    combination: dict  
        Dictionary containing hybridization, gene, registation data of the 
        current hybridization and the registerd coords of the selected peaks 
        for the current gene.
    overlapping_percentage: float 
        Percentage of overlap extracted from the Experimental_metadata.yaml
    image_size: int 
        Size of the image (consider a square image)
    plx: int 
        Radius of pixel used to create the nhood.
    
    
    Returns:
    ---------

    all_coords_cleaned_dict: dict 
        Dictionary with the dots coords of the processed gene before and
        after removal plus the coords of the removed dots. 
    
    """
    # Determine hyb and gene processed
    gene = combination['gene']
    hybridization = combination['hybridization']
    
    micData = combination['reg_data_combination']['micData']
    
    # create the dictionaries
    all_removed=dict()
    all_coords_cleaned_dict=dict()
    all_removed[hybridization]={}
    all_coords_cleaned_dict[hybridization]={}

    all_removed[hybridization][gene]={}
    
    # Calculate all the neighborhood where to remove the dots
    all_nhood = overlapping_images_grid(micData)
    
    
    for idn,image_list in enumerate(all_nhood):
        overlapping_result_dict=catch_duplicated_dots(combination, image_list,
                          overlapping_percentage,image_size,pxl)
        all_removed[hybridization][gene][idn]=overlapping_result_dict
    
    all_coords_cleaned,removed_coords,all_coords = all_dots_removal(combination,all_removed)
    all_coords_cleaned_dict[hybridization][gene]={'all_coords_cleaned':all_coords_cleaned,
                                        'removed_coords':removed_coords,
                                        'all_coords':all_coords}
    
    return all_coords_cleaned_dict

def combine_raw_counting_results(flt_rawcnt_config,hybridizations_infos,
                                 experiment_infos,processing_experiment_directory,
                                stitched_reference_files_dir,reference_gene,add_slash):

    # Collect all the counts in a dictionary maintaining the pos references

    # Prepare the dictionary where to store the raw counts
    all_raw_counts = dict()
    all_raw_counts['selected_peaks_coords_aligned'] = dict()
    all_raw_counts['selected_peaks_coords_not_aligned'] = dict()
    all_raw_counts['remaining_data'] = dict()
    all_raw_counts['registration_data'] = dict()
    skip_genes_counting=flt_rawcnt_config['skip_genes_counting']
    skip_tags_counting=flt_rawcnt_config['skip_tags_counting']

    for hybridization in hybridizations_infos.keys():

        all_raw_counts['selected_peaks_coords_aligned'][hybridization] = dict()
        all_raw_counts['selected_peaks_coords_not_aligned'][hybridization] = dict()
        all_raw_counts['remaining_data'][hybridization]= dict()

        hyb_short = re.sub('Hybridization','hyb',hybridization)
        processing_hyb = experiment_infos['ExperimentName']+'_'+hyb_short
        hyb_dir = processing_experiment_directory+processing_hyb+add_slash

        # Determine the genes to stitch in the processing hybridization
        genes_processing = hybridizations_infos[hybridization].keys()

        # Determine the counting directories
        counting_gene_dirs_path = hyb_dir+flt_rawcnt_config['analysis_name']+'_'+processing_hyb +'_counting'+add_slash
        counting_gene_dirs = glob.glob(counting_gene_dirs_path+'*')

        # Load the _reg_data.pkl files with the information about the registration
        reg_fname = stitched_reference_files_dir + flt_rawcnt_config['analysis_name'] +'_'+ \
                                            processing_hyb+'_'+reference_gene+'_stitching_data_reg.pkl'
        reg_data = pickle.load(open(reg_fname,'rb'))


        all_raw_counts['registration_data'][hybridization]= reg_data

        for gene in genes_processing:
            if gene not in skip_genes_counting or [tag for tag in skip_tags_counting if tag in gene]:
                all_raw_counts['selected_peaks_coords_aligned'][hybridization][gene] = dict()
                all_raw_counts['selected_peaks_coords_not_aligned'][hybridization][gene] = dict()
                all_raw_counts['remaining_data'][hybridization][gene]= dict()


                # Determine the counting directory to process
                counting_files_dir = [pkl_dir for pkl_dir in counting_gene_dirs if gene in pkl_dir][0]
                counting_files_dir = counting_files_dir +add_slash

                # Get list of the files with counts
                counting_files = glob.glob(counting_files_dir+'*.pkl')


                # Load the raw counting data and combine them into a dictionary
                for counting_file in counting_files:
                    pos = int(counting_file.split('_')[-1].split('.')[0])
                    data = pickle.load(open(counting_file,'rb'))

                    all_raw_counts['selected_peaks_coords_not_aligned'][hybridization][gene][pos] = data['selected_peaks']
                    all_raw_counts['remaining_data'][hybridization][gene][pos] = dict()
                    for key in data.keys():
                        if key != 'selected_peaks':
                            all_raw_counts['remaining_data'][hybridization][gene][pos][key] = data[key]

                # Recalculate the coords but keep position reference
                all_raw_counts = register_dots_coords(reg_data,hybridization,gene,all_raw_counts)


    # Save all the data
    counting_data_name = processing_experiment_directory +experiment_infos['ExperimentName']+'_all_raw_counting_data.pkl'
    pickle.dump(all_raw_counts,open(counting_data_name,'wb'))
    
    return all_raw_counts
