import numpy as np
from sympy import Point, Line, Segment
from skimage import filters,io,img_as_float,exposure,morphology,segmentation,measure,feature
from scipy import ndimage as nd
import itertools


def image_chunks_calculator(dimension,chunk_size):
    """
    Helper function to calculate the size of the chunks created according
    the length of the vector and the chunk size.

    Parameters:
    -----------

    dimension: int
        Length of the vector to Chunk
    chunkSize: int 
        Dimension of the Chunks

    Returns:
    -----------

    chunks_sizes: np.array 
        Array of the sizes of the created chunks. It deals with conditions 
        when the expected chunks size do not fit an even number of times in the 
        dimension
    """
    number_even_chunks=int(dimension//chunk_size)
    total_size_even_chunks=number_even_chunks*chunk_size
    odd_tile_size=dimension-total_size_even_chunks
    chunk_sizes=[]
    chunks_sizes=list(np.repeat(chunk_size,number_even_chunks-1))
    if odd_tile_size < chunk_size:
        chunks_sizes.append(chunk_size+odd_tile_size)
    else:
        chunks_sizes.append(odd_tile_size)
    return tuple(chunks_sizes)



def image_chunking(img,chunk_size,percent_padding):
    """
    Function used to generate the coords of the images according to the
    chunking 

    Parameters:
    -----------

    stitched_file_hdl: pointer 
        Handle of the hdf5 file that contains the stitched image.
    gene: str 
        Processed gene ('Nuclei').
    PercentPadding: float 
        Percent of overlapping between the different images (Ex. 0.2).
    ChunkSize: int 
        Dimension of the Chunks.

    Returns:
    -----------

    Coords_Chunks_list: list 
        List of np.array with the coords of the images without padding
    Coords_Padded_Chunks_list: list 
        List of np.array with the coords of the images with padding
    
    Notes:
    ------

    For both lists each np.array contains the coords in the following order:
    [row_tl,row_br,col_tl,col_br]

    """
#     img_r,img_c=stitched_file_hdl[gene][FolderLevel]['final_image'].shape
    
    img_r,img_c=img.shape
    pixel_padding = int(chunk_size*percent_padding)


    # Calculate the size of the chunks
    r_chunks_size = image_chunks_calculator(img_r,chunk_size)
    c_chunks_size = image_chunks_calculator(img_c,chunk_size)
    # Calculate the total numbers of chunks
    nr_chunks = len(r_chunks_size)
    nc_chunks = len(c_chunks_size)


    # Coords top left corner (tl)
    r_coords_tl = np.arange(0,chunk_size*(nr_chunks),chunk_size)
    c_coords_tl = np.arange(0,chunk_size*(nc_chunks),chunk_size)
    # Coords of all the tl in the image
    r_coords_tl_all,c_coords_tl_all = np.meshgrid(r_coords_tl,c_coords_tl,indexing='ij')

    # Calculate all the br coords
    r_coords_br_all = r_coords_tl_all.copy()
    c_coords_br_all = c_coords_tl_all.copy()

    for c in np.arange(0,r_coords_tl_all.shape[1]):
        r_coords_br_all[:,c] = r_coords_br_all[:,c]+r_chunks_size

    for r in np.arange(0,r_coords_tl_all.shape[0]):
         c_coords_br_all[r,:] = c_coords_br_all[r,:]+c_chunks_size

    # Calculate the padded coords
    r_coords_tl_all_padded = r_coords_tl_all-pixel_padding
    c_coords_tl_all_padded = c_coords_tl_all-pixel_padding
    r_coords_br_all_padded = r_coords_br_all+pixel_padding
    c_coords_br_all_padded = c_coords_br_all+pixel_padding

    # Correct for coords out of the image (where tl<0,br>Img.shape)
    r_coords_tl_all_padded[r_coords_tl_all_padded<0] = r_coords_tl_all[r_coords_tl_all_padded<0]
    c_coords_tl_all_padded[c_coords_tl_all_padded<0] = c_coords_tl_all[c_coords_tl_all_padded<0]
    r_coords_br_all_padded[r_coords_br_all_padded>img_r] = r_coords_br_all[r_coords_br_all_padded>img_r]
    c_coords_br_all_padded[c_coords_br_all_padded>img_c] = c_coords_br_all[c_coords_br_all_padded>img_c]

    # The coords list are generated as:
    # row_tl,row_br,col_tl,col_br

    # Create a list for the non padded coords
    Coords_Chunks_list = list()
    for r in np.arange(0,r_coords_tl_all.shape[0]):
        for c in np.arange(0,r_coords_tl_all.shape[1]):
            Coords_Chunks_list.append((np.array([r_coords_tl_all[r][c],\
                                                 r_coords_br_all[r][c],\
                                                 c_coords_tl_all[r][c],\
                                                 c_coords_br_all[r][c]])))

    # Create a list for the padded coords
    Coords_Padded_Chunks_list = list()
    for r in np.arange(0,r_coords_tl_all_padded.shape[0]):
        for c in np.arange(0,r_coords_tl_all_padded.shape[1]):
            Coords_Padded_Chunks_list.append(np.array([r_coords_tl_all_padded[r][c],\
                                                       r_coords_br_all_padded[r][c],\
                                                       c_coords_tl_all_padded[r][c],\
                                                       c_coords_br_all_padded[r][c]]))

    return nr_chunks,nc_chunks,Coords_Chunks_list, Coords_Padded_Chunks_list,r_coords_tl_all_padded,\
            c_coords_tl_all_padded,r_coords_br_all_padded,c_coords_br_all_padded

def polyT_segmentation(Img, min_object_size, min_distance,disk_radium_rank_filer,threshold_rel,trimming):
    
    """
    Function that runs the polyT segmentation on an image

    Parameters:
    -----------

    Img: np.array
        2D image to be segmented
    min_object_size: int
        Minimum size of a connected component in order to be considered a cell.
    min_distance: int
        Minimum distance between peaks for the creation of the markers for the
        first round of watershed.
    disk_radium_rank_filer: int
        Size of the filter for preprocessing the image in order to calculate the
        Thr.
    threshold_rel: float
        relative threshold between the markers for the first round of watershed.
    trimming: int 
        Define the trimming point for the selection of the Thr for image 
        processing.


    Returns:
    --------

    Objects_dict: dict
        Dictionary with the labels and their coords.

    """


    Img = img_as_float(Img)
    Hist=exposure.histogram(Img)
    ThrArray=Hist[1]
    SelectedThr = ThrArray[trimming]
    
    # Consider the case with a black image
    Objects_dict=dict()
    if Img.max()==0:
        Objects_dict={}
    else:

    #     SelectedThr=rescaling_calculation(Img,relax)
        Img=exposure.rescale_intensity(Img,in_range=(0,SelectedThr))
        Img_Thr=filters.rank.maximum(Img,morphology.disk(disk_radium_rank_filer))
        # If ranking filter remove all the objects the image is made of all 0
        # and otsu won't work

        if Img_Thr.max()==0:
            Objects_dict={}
        else:

            Threshold=filters.threshold_otsu(Img_Thr)
            Img_Thr=Img_Thr>Threshold

            # Fill holes in the binarized objects
            ImgFill=nd.morphology.binary_fill_holes(Img_Thr)
            # Binary opening
            Img_Thr_op=nd.binary_opening(ImgFill,structure=morphology.disk(2))
            # Remove small object
            ImgRemSmall=morphology.remove_small_objects(Img_Thr_op,min_size=min_object_size)

            # Determine if by removing the small objects you fully cleared the image
            Check = np.asarray(ImgRemSmall, dtype=int)
            Check = Check.sum()
            if Check>0:
                # Identify the objects 
                labels_peak=measure.label(ImgRemSmall,connectivity=2,background=0)

                # Create markers for the first round of watersher
                distance=nd.distance_transform_edt(Img*ImgRemSmall)

                local_maxi = feature.peak_local_max(distance,min_distance=min_distance,threshold_rel=threshold_rel,\
                                                    indices=False,labels=labels_peak,exclude_border = False)

                # Take care of the case where there is an object but no local max and
                # cannot determine the number of values
                if local_maxi.max()>0:
                    marker=markers = nd.label(local_maxi)[0]
                    # Remember that background is 0
                    Objects_wt_one = morphology.watershed(-distance, marker, mask=ImgRemSmall)
                    # no need for intensity...need to be recalculate lated
                    #Objects_wt_one_prop=measure.regionprops(Objects_wt_one,intensity_image=Img)
                    Objects_wt_one_prop=measure.regionprops(Objects_wt_one)

                    # Select only newly created objects above minsize 
                    for obj in Objects_wt_one_prop:
                        if obj.area> min_object_size:
                            # Return only the labels and the corresponding coords
                            Objects_dict[obj.label]=obj.coords
                else:
                    Objects_dict={}
            else:
                Objects_dict={}
    return Objects_dict

def OverlappingCouples(chunk_coords,TotalDataDict):
    """
    Calculate all the intersecting objects
    
    Parameters:
    -----------

    chunk_coords: list 
        List of np.array with the coords of the images with padding
    TotalDataDict: dict
        Dictionary with the labels and the coords of the identified objects
        after merging the result of the segmentation on the different image
        chunks.

    Returns:
    --------

    Intersecting_cpl: list
        List of tuples with all the overlapping labels
        [(Obj1,Obj12),(Obj2, Obj3)].

    """
    # Filter out the objects not in the overlapping region
    ObjectsOverlap_list=list()
    Intersecting_cpl=list()
    Intersecting_cpl_idx=list()
    Intersection=list()
    for lab in TotalDataDict.keys():
        if ((TotalDataDict[lab][:,0] >= chunk_coords[0]).any() and
            (TotalDataDict[lab][:,0] <= chunk_coords[1]).any() and
            (TotalDataDict[lab][:,1] >= chunk_coords[2]).any() and
            (TotalDataDict[lab][:,1] <= chunk_coords[3]).any()):
                ObjectsOverlap_list.append(lab)
    
    # Remove possible duplicates
    ObjectsOverlap_list=list(set(ObjectsOverlap_list))
    
    # Create all possible combinations of overlapping objects
    combinations=list(itertools.combinations(ObjectsOverlap_list,2))
    
    # For all the couples of objects determine which one are overlapping 
    for couple in combinations:
        Result=not set(map(tuple, TotalDataDict[couple[0]])).isdisjoint(map(tuple, TotalDataDict[couple[1]]))
        Intersection.append(Result)
    if (np.array(Intersection)).any():
        [Intersecting_cpl_idx.append(i) for i,x in enumerate(Intersection) if x]
        [Intersecting_cpl.append(combinations[cp]) for cp in Intersecting_cpl_idx]
    
    return Intersecting_cpl

# Function for calculating the intersection. Recalculate the coords
# Output the coords and the labels to keep and to get rid off
# in this case I merge the coords but in the future I will merge then recalculate
# the segmentation
def IntersectionCouples(couple,TotalDataDict):

    """
    Function to consolidate the overlapping couples. Identify if there are
    multiple overlapping, merge the coords together and keep only one label
    
    Parameters:
    -----------
    couple: list
        List of tuples with the labels of the overlapping objects. If contains
        more than one repeat of the same object, the repeats are removed.
    TotalDataDict: dict
        Dictionary with the labels and the coords of the identified objects
        after merging the result of the segmentation on the different image
        chunks.   

    Returns:
    --------
    SaveLab: list
        List of saved labels.
    RemoveLabs: list
        List of the remove labels.
    NewCoords: list
        Merged obj coords.

    """
    # If there are more then one combination in the couple
    if len(couple)>1:
        # Remove the duplicated
        labs=list()
        [labs.append(element) for cpl in couple for element in cpl]
        labs=list(set(labs))
    else:
        labs=list(couple[0])
    # Merge the coords
    NewCoords=TotalDataDict[labs[0]]
    for lab in labs[1:]:
        NewCoords=np.vstack((NewCoords,TotalDataDict[lab]))

    # Remove duplicates
    Tp_NewCoord=[tuple(x) for x in NewCoords]
    NewCoords=np.array(list(set(Tp_NewCoord)))

    TotalDataDict[labs[0]]=NewCoords
    
    SaveLab=labs[0]
    RemoveLabs=labs.copy()
    RemoveLabs.remove(SaveLab)
    
    return SaveLab, RemoveLabs,NewCoords 

def obj_properties_calculator(obj):

    """
    Calculate the morphological properties of the object (connected component).
    
    Parameters:
    -----------

    obj: dict
        Dictionary with the object label and the coords.

    Returns:
    --------
    
    obj_prop_dict: dict
        Dictionary with the calculated properties of the object


    """
    
    # Create the obj properties dictionary
    obj_prop_dict = dict()
    
    # Define ID and coords of the obj
    obj_id = obj[0] 
    obj_coords = obj[1]
    
    row_Coords_obj=obj_coords[:,0]
    col_Coords_obj=obj_coords[:,1]
    
    # Calculate the coords of the bounding box to use to crop the ROI
    # Added +1 because in the cropping python will not consider the end of the interval
    bb_row_max=row_Coords_obj.max()+1
    bb_row_min=row_Coords_obj.min()
    bb_col_max=col_Coords_obj.max()+1
    bb_col_min=col_Coords_obj.min()
    
    
    # Normalize the coords of the obj
    row_Coords_obj_norm=row_Coords_obj-bb_row_min
    col_Coords_obj_norm=col_Coords_obj-bb_col_min
    
    
    # Bounding box
    bb_coords = (bb_row_min,bb_row_max,bb_col_min,bb_col_max)
    
    # obj_coords recalculated
    obj_coords = (row_Coords_obj_norm,col_Coords_obj_norm)
    
    
    # Create object image for properties
    Img=np.zeros([bb_coords[1]-bb_coords[0],bb_coords[3]-bb_coords[2]],dtype=np.uint8)
    Img[obj_coords[0],obj_coords[1]]=1

    # calculate the object properties
    obj_prop=measure.regionprops(Img)

    # Collect only the needed properties
    obj_area=obj_prop[0].area
    obj_centroid=obj_prop[0].centroid
    obj_convex_area=obj_prop[0].convex_area
    obj_filled_area=obj_prop[0].filled_area
    obj_major_axis_length=obj_prop[0].major_axis_length
    obj_minor_axis_length=obj_prop[0].minor_axis_length
    obj_perimeter=obj_prop[0].perimeter

    # Adjust coords of the centroid
    obj_centroid=[obj_centroid[0]+bb_coords[0],obj_centroid[1]+bb_coords[2]]
  
    obj_prop_dict[obj_id]={'obj_area':obj_area,'obj_centroid':obj_centroid,'obj_convex_area':obj_convex_area,\
                        'obj_filled_area':obj_filled_area,'obj_major_axis_length':obj_major_axis_length,\
                        'obj_minor_axis_length':obj_minor_axis_length,'obj_perimeter':obj_perimeter,
                        'obj_coords':obj[1]}
    
    return obj_prop_dict
