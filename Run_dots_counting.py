
# Check the imports that are not required

# General imports
import argparse
from mpi4py import MPI
import glob
import h5py
from skimage import filters,io,img_as_float,exposure,morphology,segmentation,measure,feature,color,img_as_ubyte
import os
from scipy import ndimage as nd
import glob
import re
import joblib
import numpy as np
import itertools
import shutil

# Own imports
from pysmFISH import utils
from pysmFISH import MPI_utils
from pysmFISH import dots_calling
from pysmFISH import h5writingfun

# ---------------- Functions --------------------------

def crop_object_coords(data_file_hdl,obj,segmentation_gene):
    
    # Coords of the obj
    row_Coords_obj=data_file_hdl[analysis_name]['Segmentation'][segmentation_gene]['Objects'][obj][:,0]
    col_Coords_obj=data_file_hdl[analysis_name]['Segmentation'][segmentation_gene]['Objects'][obj][:,1]
    
    # Calculate the coords of the bounding box to use to crop the ROI
    # Added +1 because in the cropping python will not consider the end of the interval
    bb_row_max=row_Coords_obj.max()+1
    bb_row_min=row_Coords_obj.min()
    bb_col_max=col_Coords_obj.max()+1
    bb_col_min=col_Coords_obj.min()
    
    # Normalize the coords of the obj
    row_Coords_obj_norm=row_Coords_obj-bb_row_min
    col_Coords_obj_norm=col_Coords_obj-bb_col_min
    
    
    return (bb_row_min,bb_row_max,bb_col_min,bb_col_max),(row_Coords_obj_norm,col_Coords_obj_norm)

# Check on how you can merge dictionary
# datatype non used but necessary in order to avoid a data type error in the 
# implementation of MPI
def addCounter(counter1, counter2, datatype):
    for item in counter2:
            counter1[item] = counter2[item]
    return counter1


# ---------------- MPI parameters --------------------------
comm = MPI.COMM_WORLD
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process


# # input
parser = argparse.ArgumentParser(description='Script counting dots')
parser.add_argument('-d', help='Experimental directory')
parser.add_argument('-n',help='Hybridization Number')
parser.add_argument('-f',help='Preprocessing level (ex. StitchedImageRegistered)')
parser.add_argument('-x',help='AnalysisName')
parser.add_argument('-g',help='segmentation gene')


args= parser.parse_args()
data_folder=args.d
hyb_nr=int(args.n)
FolderLevel=args.f
analysis_name=args.x
segmentation_gene=args.g
# inputs for testing
# analysis_name='AnalysisNovember2016'
# hyb_nr=1
# segmentation_gene='polyA'
# Folder containing the image to use for counting
# FolderLevel='StitchedImageRegistered'
# data_folder='/data/users/simonec/Data_Simone/smFISH/Cortex/Stitched/'

skipping=['Nuclei','polyA']



# ------------- File Loading --------------------------------
# # Open the the file with the hybridization to count
stitched_file_name = glob.glob(data_folder + '*_Hybridization' +
                                str(hyb_nr) + '.sf.hdf5')[0]

# # for testing
# stitched_file_name='/data/users/simonec/Data_Simone/smFISH/Cortex/Stitched/AnalysisNovember2016_Hybridization1.sf.hdf5'

stitched_file_hdl=h5py.File(stitched_file_name,'r',driver='mpio',comm=MPI.COMM_WORLD)

# ------------- File Analysis --------------------------------


# data_file_name='/data/users/simonec/Data_Simone/smFISH/Cortex/Stitched/AnalysisNovember2016Analysis.data.hdf5'

data_file_name=data_folder+'Analysis.data.hdf5'
# Open in rank 0 than bcast the coords and bb_coords to the other cores


data_file_hdl=h5py.File(data_file_name,'r+',driver='mpio',comm=MPI.COMM_WORLD)

# Create list of genes to process
ToKeep=set(stitched_file_hdl)-set(skipping)
genes_list=[gene for gene in ToKeep if '_IF' not in gene]

# # Create the gene groups
# if genes_list:
#     for gene in stitched_file_hdl:
#         if gene not in ['Nuclei','polyA'] or '_IF' not in gene:
#             gene_grp=data_file_hdl[analysis_name].create_group(gene)


# if genes_list:

# IT IS WORTH TO WRITE IN PARALLEL TO SPEED UP THE PROCESS

# Try to avoid parallel writing on hdf5 so you can make it easier to move to
# a non MPI and more pythonic system

# Presave the data to avoid to used parallel hdf5....also difficult because I
# don't know the size of the datasets than need to be pre-created on all cores
# I will have to create a resizable dataset
# Create temporary folder for saving the counting
if rank==0:

    for gene in genes_list:
            DirPath=data_folder+'RNA_counting_tmp_Hybridization'+str(hyb_nr)+'_'+gene
            try:
                os.stat(DirPath)
            except:
                os.mkdir(DirPath)
                os.chmod(DirPath,0o777)


# ------------- Scatter the obj regions for counting ---------------------------
if rank==0:
    
    Objects=list(data_file_hdl[analysis_name]['Segmentation'][segmentation_gene]['Objects'].keys())

    TasksList=np.array([np.int32(obj) for obj in Objects])
     # Chunk the distribution list
    Chunked_list,NumberChunks,Displacement = MPI_utils.tasks_chunking(TasksList,size)


else:
    NumberChunks=None
    Displacement=None
    Chunked_list=None
    TasksList=None

TasksList=comm.bcast(TasksList,root=0)
# Make the cores aware of the number of jobs that will need to run
# The variable count is created by the scatter function and has the number of
# processes and is different in every core
cnt = np.zeros(1, dtype='int32')
comm.Scatter(NumberChunks, cnt, root=0)
# Define the local variable that will be filled up wuth the scattered data
xlocal = np.zeros(cnt, dtype='int32') # Use rank for determine the size of the xlocal on the different cores
# Scatter the value of tasks to the different cores
comm.Scatterv([TasksList,NumberChunks,Displacement,MPI.INT],xlocal, root=0)

# Create a coords dict on each rank
# It may not be efficient to calculate it every time an hyb is processed. I save
# in memory so will be used for all the channels in the hyb. I think that saving it
# and load it may be slower
Coords_dict=dict()
for obj in xlocal:
    bb_coords,obj_coords=crop_object_coords(data_file_hdl,str(obj),segmentation_gene)
    Coords_dict[obj]={'bb_coords':bb_coords,'obj_coords':obj_coords}

# process each gene separetely. I could also create a massive list but as 
# starting point I will process each gene
for gene in genes_list:
    tmp_folder_path=data_folder+'RNA_counting_tmp_Hybridization'+str(hyb_nr)+'_'+gene+'/'
    for obj in xlocal:
        bb_coords=Coords_dict[obj]['bb_coords']
        obj_coords=Coords_dict[obj]['obj_coords']
        
        # Load the cropped image
        Img_cropped=stitched_file_hdl[gene][FolderLevel]['final_image'][bb_coords[0]:bb_coords[1],bb_coords[2]:bb_coords[3]]
        
        # Mask the image according to the segmented region
        Img_masked=np.zeros_like(Img_cropped)
        Img_masked[obj_coords[0],obj_coords[1]]=Img_cropped[obj_coords[0],obj_coords[1]]

        Selected_Thr,Calculated_Thr,Selected_Peaks,ThrArray,PeaksCoords,\
             TotalPeaks,Thr_idx,stringency,Selected_peaks_int=dots_calling.ThrCalculator(Img_masked,min_distance=5,stringency=0,dimensions=2)
        
        
#         # Consider black image case
        if Selected_Thr>0:
#             # Intensity counting of the max peaks
#             Selected_peaks_int=Img_masked[Selected_Peaks[:,0],Selected_Peaks[:,1]]

            # Dots coords correction (in order to be able to map the dots on the)
            # stitched image and not only in the selected region.
            
            # Recalculate the coords of all the peaks (with different thresholds)
            PeaksCoords_reMapped_tmp=[]
            PeaksCoords_reMapped=[]
            for peaks_coords in PeaksCoords:
                for peak in peaks_coords:
                    NewCoords=[peak[0]+bb_coords[0],peak[1]+bb_coords[2]]
                    # NewCoords=peak+[bb_coords[0],bb_coords[2]]
                    PeaksCoords_reMapped_tmp.append(NewCoords)
                PeaksCoords_reMapped.append(PeaksCoords_reMapped_tmp)
                PeaksCoords_reMapped_tmp.append(NewCoords)
            
            # Recalculate the coords of the selected peaks
            Selected_Peaks_reMapped=[]
            for peak in Selected_Peaks:
                NewCoords=[peak[0]+bb_coords[0],peak[1]+bb_coords[2]]
                # NewCoords=peak+[bb_coords[0],bb_coords[2]]
                Selected_Peaks_reMapped.append(NewCoords)
        else:
            PeaksCoords_reMapped_tmp=PeaksCoords
            Selected_Peaks_reMapped=Selected_Peaks

        # Save the tmp file
        # Save the data in a compressed format before loading them into a hdf5_file
        joblib.dump((Selected_Thr,Calculated_Thr,Selected_Peaks_reMapped,\
            Selected_peaks_int,ThrArray,PeaksCoords_reMapped_tmp,\
            TotalPeaks,Thr_idx,stringency,bb_coords),tmp_folder_path+'Hybridization'+str(hyb_nr)+'_'+gene+'_PeaksData_obj_'+str(obj)+'.pkl')


stitched_file_hdl.close()
data_file_hdl.close()

# # Write everything on the Analysis Results file
if rank==0:
    data_file_hdl=h5py.File(data_file_name,'r+',swmr=True,libver='latest')
    SegmentedRegionsCounts_grp=data_file_hdl[analysis_name].require_group('SegmentedRegionsCounts')
    HybRef='Hybridization'+str(hyb_nr)
    Hybridization_grp=SegmentedRegionsCounts_grp.require_group(HybRef)
    for gene in genes_list:
        _=Hybridization_grp.require_group(gene)
        for obj in Objects:   
            tmp_folder_path=data_folder+'RNA_counting_tmp_Hybridization'+str(hyb_nr)+'_'+gene+'/'  
            h5writingfun.serial_writing_dots_segreg(data_file_hdl,analysis_name,tmp_folder_path,\
                hyb_nr,gene,obj)
        # delete the tmp folders
        shutil.rmtree(tmp_folder_path)

    data_file_hdl.close()

