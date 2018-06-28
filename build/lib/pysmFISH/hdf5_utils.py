"""
Functions that handle hdf5 files
"""

import h5py
import numpy as np
import glob
from collections import OrderedDict
import pickle
import loompy


def hdf5_create_preprocessing_file(hybridizations_infos,processing_hyb,hybridization,
                                   analysis_name, hyb_dir,converted_positions,
                                   image_properties):
    
    """ 

    Create the hdf5 file that will contain the preprocessed images
    for one hybridization.
    The preprocessing .hdf5 file must have a .ppf.hdf5 extension
    The file will be created for serial writing of the hdf5 file.
    After creation the file is closed.
    Remember that the 'lzf' compression do not accept extra items
    
    Parameters:
    -----------

    hybridizations_infos: dict 
        Parsed Hybridizations metadata
    processing_hyb: str
        Hybridization to process. The function will create an hdf5 file for the 
        hybridization only
    hybridization: str 
        Hybridization name
    analysis_name: str 
        Analysis name
    hyb_dir: str 
        Path of the dir where to save the file
    converted_positions: dict 
        Parsed positions
    """
    
    compression = 'lzf'
    preprocessing_file_name= analysis_name+'_'+processing_hyb+'.ppf.hdf5'
    preprocessing_file = h5py.File(hyb_dir+preprocessing_file_name, 'w',libver='latest')
    
    # ImagesInfo
    Hyb_cols_num_px=image_properties['HybImageSize']['columns']
    Hyb_rows_num_px=image_properties['HybImageSize']['rows']

    Strip_cols_num_px=image_properties['StripImageSize']['columns']
    Strip_rows_num_px=image_properties['StripImageSize']['rows']

    # Create the groups and the subgroups

    # List of the XY positions
    Positions_list=list(converted_positions[hybridization].keys())

    for gene in hybridizations_infos[hybridization].keys():
            gen_grp=preprocessing_file.create_group(gene)
            Fd_group=gen_grp.create_group('FilteredData')
            preprocessing_file.flush()
            if 'Hyb' in hybridization:

                for pos in Positions_list:
                    Fd_group.create_dataset(str(pos),dtype = np.uint16, \
                        shape=(Hyb_rows_num_px,Hyb_cols_num_px),\
                        compression=compression,\
                        shuffle=True,chunks=(Hyb_rows_num_px,Hyb_cols_num_px))
            else:
                for pos in Positions_list:
                    Fd_group.create_dataset(str(pos),dtype = np.uint16,\
                        shape=(Strip_rows_num_px,Strip_cols_num_px),\
                        compression=compression,\
                        shuffle=True,chunks=(Hyb_rows_num_px,Hyb_cols_num_px))

    preprocessing_file.close()
    preprocessing_file_path=hyb_dir + preprocessing_file_name
    
    return preprocessing_file_path




def create_loompy_raw_counting(hybridizations_infos,converted_positions,
                        hybridization,flt_rawcnt_config,hyb_dir,
                        processing_hyb,counting_gene_dirs):
    """
    Function used to write the counting results in a loom file

    Parameters:
    -----------
    
    hybridizations_infos: dict 
        Parsed information on the experiment.
    converted_positions: dict 
        coords of the images for all hybridization.
    hybridization: str 
        hybridization processed (ex. Hybridization2) 
    flt_rawcnt_config: dict
        Parsed filtering a raw counting configuration file 
        (Filtering_raw_counting.config.yaml)
    counting_gene_dirs: list 
        List of the directories containing the counts


    """

    # Loompy matrix column attributes
    pos_att = np.arange(0,len(converted_positions[hybridization].keys()))

    # Create dictionaries to store the data that have one array for each image
    # position

    counting_dict = OrderedDict()
    selected_peaks_dict = OrderedDict()
    thr_array_dict = OrderedDict()
    peaks_coords_dict = OrderedDict()
    total_peaks_dict = OrderedDict()
    selected_peaks_int_dict = OrderedDict()


    gene_idx_pos = 0
    gene_idx_list = []
    gene_list = list(hybridizations_infos[hybridization].keys())
    gene_list = [gene for gene in gene_list if gene not in flt_rawcnt_config['skip_genes_counting'] ]
    gene_list = [gene for tag in flt_rawcnt_config['skip_tags_counting'] for gene in gene_list if tag not in gene]

    # Create matrices for loompy layers
    total_counts_mat = np.zeros([len(gene_list),len(converted_positions[hybridization].keys())])
    selected_thr_mat = np.zeros([len(gene_list),len(converted_positions[hybridization].keys())])
    calculated_thr_mat = np.zeros([len(gene_list),len(converted_positions[hybridization].keys())])
    thr_idx_mat = np.zeros([len(gene_list),len(converted_positions[hybridization].keys())])
    stringency_mat = np.zeros([len(gene_list),len(converted_positions[hybridization].keys())])

    array_positions = np.sort(list(converted_positions[hybridization].keys()))

    if gene_list:
        for gene in gene_list:
            gene_idx_list.append(gene+'_'+hybridization)
            counting_gene_dir = [fdir for fdir in counting_gene_dirs if gene in fdir][0]
            counting_files_list = glob.glob(counting_gene_dir+'*.pkl')

            counting_dict[gene] =OrderedDict()

            counting_dict[gene]['selected_peaks'] =OrderedDict()
            counting_dict[gene]['thr_array'] =OrderedDict()
            counting_dict[gene]['peaks_coords'] =OrderedDict()
            counting_dict[gene]['total_peaks'] =OrderedDict()
            counting_dict[gene]['selected_peaks_int'] =OrderedDict()



            # Process the files according to sorted position
            for pos in array_positions:
                counting_file = [cf for cf in counting_files_list if 'pos_'+str(pos) in cf][0]
                countings = pickle.load(open(counting_file,'rb'))
                # pos = np.int(counting_file.split('/')[-1].split('_')[-1].split('.')[0])

                # loompy layers construction
                selected_thr_mat[gene_idx_pos,pos] = countings['selected_thr']
                calculated_thr_mat[gene_idx_pos,pos] = countings['calculated_thr']
                thr_idx_mat[gene_idx_pos,pos] = countings['thr_idx']
                stringency_mat[gene_idx_pos,pos] = countings['stringency']
                if isinstance(countings['selected_peaks'], list):
                    total_counts_mat[gene_idx_pos,pos] = len(countings['selected_peaks'])
                else:
                    total_counts_mat[gene_idx_pos,pos] = 0


                # Dictionaries construction

                counting_dict[gene]['selected_peaks'][pos] = countings['selected_peaks']
                counting_dict[gene]['thr_array'][pos] = countings['thr_array']
                counting_dict[gene]['peaks_coords'][pos] = countings['peaks_coords']
                counting_dict[gene]['total_peaks'][pos] = countings['total_peaks']
                counting_dict[gene]['selected_peaks_int'][pos] = countings['selected_peaks_int']



            gene_idx_pos +=1

    # Create loompy file
    loom_fname = hyb_dir+processing_hyb+'_raw_counting.loom'
    loom_hdl = loompy.create(filename=loom_fname,matrix=total_counts_mat,row_attrs={'genes':np.array(gene_idx_list)},col_attrs={'image_pos':np.array(pos_att)})
    loom_hdl.set_layer(name='selected_thr',matrix=selected_thr_mat)
    loom_hdl.set_layer(name='calculated_thr',matrix=calculated_thr_mat)
    loom_hdl.set_layer(name='thr_idx',matrix=thr_idx_mat)
    loom_hdl.set_layer(name='stringency',matrix=stringency_mat)
    loom_hdl.close()    

    # Add extra data to the loom file
    with h5py.File(loom_fname,'r+') as loom_hdl:
        counting_arrays_grp = loom_hdl.create_group('counting_arrays')
        for gene in counting_dict.keys():
            gene_grp = counting_arrays_grp.create_group(gene+'_'+hybridization)
            selected_peaks_grp = gene_grp.create_group('selected_peaks')
            thr_array_grp = gene_grp.create_group('thr_array')
            peaks_coords_grp = gene_grp.create_group('peaks_coords')
            total_peaks_grp = gene_grp.create_group('total_peaks')
            selected_peaks_int_grp = gene_grp.create_group('selected_peaks_int')
            for pos in array_positions:
                selected_peaks_grp.create_dataset(str(pos), data=counting_dict[gene]['selected_peaks'][pos])
                thr_array_grp.create_dataset(str(pos),data=counting_dict[gene]['thr_array'][pos])
                total_peaks_grp.create_dataset(str(pos),data=counting_dict[gene]['total_peaks'][pos])
                selected_peaks_int_grp.create_dataset(str(pos),data=counting_dict[gene]['selected_peaks_int'][pos])
                pos_sbgrp = peaks_coords_grp.create_group(str(pos))
                if isinstance(counting_dict[gene]['peaks_coords'][pos], list):
                    for idx,peaks in enumerate(counting_dict[gene]['peaks_coords'][pos]):
                        pos_sbgrp.create_dataset(str(idx),data=peaks)
                else:
                    pos_sbgrp.create_dataset(str(idx),data=peaks)