import pickle
import numpy as np
from skimage import io, img_as_float
from .dots_calling import thr_calculator
from .filtering import smFISH_filtering, nuclei_filtering



def filtering_and_counting(fpath_img_to_filter,filtered_png_img_gene_dirs,filtered_img_gene_dirs,
        counting_gene_dirs, illumination_correction=False ,plane_keep=None, min_distance=5, stringency=0,
        skip_genes_counting=None,skip_tags_counting=None):
    """
    Function used to clean the images and to count the smFISH dots.
    It is designed to process in parallel all the tmp file images
    stored as numpy arrays after conversion from the microscope format.
    
    Parameters:
    ------------

    fpath_img_to_filter: str 
        path to the file to process
    filtered_png_img_gene_dirs: list 
        list of the paths of the directories where the filtered images as are
        saved as pngs.
    filtered_img_gene_dirs: list 
        list of the paths of the directories where the filtered images are saved 
        as .npy.
    counting_gene_dirs: list 
        list of the paths of the directories where the countings of the filtered
        images are saved.
    illumination_correction: bool 
        if True the illumination correction is run on the dataset.
    plane_keep: list 
        start and end point of the z-planes to keep. Default None
        keep all the planes (ex. [2,-3]).
    min_distance: int 
        minimum distance between dots.
    stringency: int 
        stringency use to select the threshold used for counting.
    skip_genes_counting: list 
        list of the genes to skip for counting count.
    skip_tags_counting: list 
        list of the tags inside the genes/stainings name to avoid to count.
    """
    
    # Get infos from file name
    fname_split = fpath_img_to_filter.split('/')[-1].split('_')
    experiment_name = fname_split[0]
    hyb = fname_split[1]
    gene = fname_split[2]
    pos = fname_split[4].split('.')[0]
    
    
    # Load the image to process
    img_stack = np.load(fpath_img_to_filter) # image is np.uint16
    img_stack = img_as_float(img_stack)

    # Remove extra planes. As it is for now this step is mainly for single image
    # usage. I will include the automatic excess planes remove function to use
    # for large scale image analysis later on
    if isinstance(plane_keep,list):
        img_stack = img_stack[plane_keep[0]:plane_keep[1],:,:]


    # Filtering image according to gene
    if gene in skip_genes_counting or [tag for tag in skip_tags_counting if tag in gene]:
        # Remove the background from the nuclei
        img_filtered = nuclei_filtering(img_stack)
        counting_dict = None

    else:
        # Remove background and enhance smFISH signal
        img_filtered=smFISH_filtering(img_stack)

        # Count the dots in the whole image
        counting_dict=thr_calculator(img_filtered,min_distance,stringency)


    # Convert image to uint16
    # Clip the values above 1
    img_filtered[img_filtered>1] = 1
    # Scale to the max of the uint16
    img_filtered *= np.iinfo(np.uint16).max
    # Round and convert to integer
    img_filtered = np.uint16(np.rint(img_filtered))
    
    # Save images and dictionary
    # This part may be removed from the function in case we will run
    # temporary storage in RAM in order to reduce i/o to the common
    # HD of the cluster
    # Identify the directory for storing the images and the counting
    img_saving_dir_npy=[saving_dir for saving_dir in filtered_img_gene_dirs if gene in saving_dir.split('/')[-2] ][0]

    img_saving_dir_png=[saving_dir for saving_dir in filtered_png_img_gene_dirs if gene in saving_dir.split('/')[-2] ][0]
    

    # Save the images and the counting if performed
    fname_png = img_saving_dir_png+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.png'
    io.imsave(fname_png,img_filtered)

    fname_npy = img_saving_dir_npy+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.npy'
    np.save(fname_npy,img_filtered,allow_pickle=False)


    if counting_dict:
        # may missing if I don't want the counting
        counting_saving_dir=[saving_dir for saving_dir in counting_gene_dirs if gene in saving_dir.split('/')[-2] ][0]
        fname = counting_saving_dir+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.pkl'
        pickle.dump(counting_dict,open(fname,'wb'))
           
    return


def filtering_and_counting_experiment(fpath_img_to_filter,filtered_dir_path,
                                      counting_dir_path,exp_name,add_slash,plane_keep=None,
                                      min_distance=5,stringency=0):
    
    """
    Function to filter and count dots in the images generated from a small
    experiment.

    Parameters:
    ------------

    fpath_img_to_filter: str 
        path to the file to process.
    filtered_dir_path: list 
        list of the paths of the directories where the filtered images are saved.
    counting_dir_path: list 
        list of the paths of the directories where the counting of filtered 
        images are stored.
    exp_name: str 
        name of the experiment to process.
    plane_keep: list 
        start and end point of the z-planes to keep. Default None
        keep all the planes (ex. [2,-3]).
    min_distance: int 
        minimum distance between dots.
    stringency: int 
        stringency use to select the threshold used for counting.
    
    """

    # Load the image to process
    img_stack = np.load(fpath_img_to_filter) # image is np.uint16
    img_stack = img_as_float(img_stack)

    # Remove extra planes. As it is for now this step is mainly for single image
    # usage. I will include the automatic excess planes remove function to use
    # for large scale image analysis later on
    if plane_keep:
        img_stack = img_stack[plane_keep[0]:plane_keep[1],:,:]

    channel = fpath_img_to_filter.split(add_slash)[-1].split('_')[-3]
    fov = fpath_img_to_filter.split(add_slash)[-1].split('_')[-1].split('.')[0]
    
    not_counting=['Nuclei','Dapi','DAPI']
    
    # Filtering image according to gene
    if channel in not_counting or '_IF' in channel or channel == 'polyA':
        # Remove the background from the nuclei
        img_filtered = nuclei_filtering(img_stack)
        counting_dict = None

    else:
        # Remove background and enhance smFISH signal
        img_filtered=smFISH_filtering(img_stack)

        # Count the dots in the whole image
        counting_dict=thr_calculator(img_filtered,min_distance,stringency)
    
    
    # Convert image to uint16
    # Clip the values above 1
    img_filtered[img_filtered>1] = 1
    # Scale to the max of the uint16
    img_filtered *= np.iinfo(np.uint16).max
    # Round and convert to integer
    img_filtered = np.uint16(np.rint(img_filtered))
    
    fname = fpath_img_to_filter.split(add_slash)[-1][:-4]

    fname_path_png = filtered_dir_path+add_slash+exp_name+'_'+fname+'.png'
    io.imsave(fname_path_png,img_filtered)
    if counting_dict:
            fname_path_pkl = counting_dir_path+add_slash+exp_name+'_'+fname+'.pkl'
            pickle.dump(counting_dict,open(fname_path_pkl,'wb'))
           
    return


def filtering_and_counting_ill_correction(fpath_img_to_filter,illumination_function, filtered_png_img_gene_dirs,filtered_img_gene_dirs,
        counting_gene_dirs, illumination_correction=False ,plane_keep=None, min_distance=5, stringency=0,
        skip_genes_counting=None,skip_tags_counting=None):
    """
    Function used to clean the images and to count the smFISH dots.
    Designed to work in parallel processing all the tmp file images
    stored as numpy arrays after conversion from the microscope format.
    
    Parameters:
    ------------
    
    fpath_img_to_filter: str 
        path to the file to process.
    illumination_function: np.array float64 
         illumination function.
    filtered_png_img_gene_dirs: list 
        list of the paths of the directories where the filtered images as are
        saved as pngs.
    filtered_img_gene_dirs: list 
        list of the paths of the directories where the filtered images are saved 
        as .npy.
    counting_gene_dirs: list 
        list of the paths of the directories where the countings of the filtered
        images are saved.
    illumination_correction: bool 
        if True the illumination correction is run on the dataset.
    plane_keep: list 
        start and end point of the z-planes to keep. Default None
        keep all the planes (ex. [2,-3]).
    min_distance: int 
        minimum distance between dots.
    stringency: int 
        stringency use to select the threshold used for counting.
    skip_genes_counting: list 
        list of the genes to skip for counting count.
    skip_tags_counting: list 
        list of the tags inside the genes/stainings name to avoid to count.

    """
    
    # Get infos from file name
    fname_split = fpath_img_to_filter.split('/')[-1].split('_')
    experiment_name = fname_split[0]
    hyb = fname_split[1]
    gene = fname_split[2]
    pos = fname_split[4].split('.')[0]
    
    
    # Load the image to process
    img_stack = np.load(fpath_img_to_filter) # image is np.uint16
    img_stack = img_as_float(img_stack)

    # Remove extra planes. As it is for now this step is mainly for single image
    # usage. I will include the automatic excess planes remove function to use
    # for large scale image analysis later on
    if isinstance(plane_keep,list):
        img_stack = img_stack[plane_keep[0]:plane_keep[1],:,:]


    # Correct for illumination
    img_stack = img_stack/illumination_function

    # Filtering image according to gene
    if gene in skip_genes_counting or [tag for tag in skip_tags_counting if tag in gene]:
        # Remove the background from the nuclei
        img_filtered = nuclei_filtering(img_stack)
        counting_dict = None

    else:
        # Remove background and enhance smFISH signal
        img_filtered=smFISH_filtering(img_stack)

        # Count the dots in the whole image
        counting_dict=thr_calculator(img_filtered,min_distance,stringency)
        
    # Convert image to uint16
    # Clip the values above 1
    img_filtered[img_filtered>1] = 1
    # Scale to the max of the uint16
    img_filtered *= np.iinfo(np.uint16).max
    # Round and convert to integer
    img_filtered = np.uint16(np.rint(img_filtered))
    
    # Save images and dictionary
    # This part may be removed from the function in case we will run
    # temporary storage in RAM in order to reduce i/o to the common
    # HD of the cluster
    # Identify the directory for storing the images and the counting
    img_saving_dir_npy=[saving_dir for saving_dir in filtered_img_gene_dirs if gene in saving_dir.split('/')[-2] ][0]

    img_saving_dir_png=[saving_dir for saving_dir in filtered_png_img_gene_dirs if gene in saving_dir.split('/')[-2] ][0]

    

    # Save the images and the counting if performed
    fname_png = img_saving_dir_png+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.png'
    io.imsave(fname_png,img_filtered)

    fname_npy = img_saving_dir_npy+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.npy'
    np.save(fname_npy,img_filtered,allow_pickle=False)


    if counting_dict:
        # may missing if I don't want the counting
        counting_saving_dir=[saving_dir for saving_dir in counting_gene_dirs if gene in saving_dir.split('/')[-2] ][0]
        fname = counting_saving_dir+experiment_name+'_'+hyb+'_'+gene+'_'+'pos_'+pos+'.pkl'
        pickle.dump(counting_dict,open(fname,'wb'))
           
    return