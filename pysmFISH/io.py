
import numpy as np
import nd2reader as nd2

def nd2_to_npy(gene,raw_data_gene_fname,tmp_gene_dir,processing_hyb,use_ram=True,max_ram=150000000000):
       
    """    
    Function used to convert Nikon ND2 files in numpy arrays.
    Important: require the nd2reader (https://github.com/jimrybarski/nd2reader)
    pip  install "nd2reader==2.1.3". There is a new version based on pims
    but need to be tested
    The input ND2 file will have only one channel and will be converted in a
    binary npy.
    
    Parameters:
    -----------

    gene: str 
        Name of the gene to process (ex. Aldoc)
    raw_data_gene_fname: str
        Path to the ND2 file (ex. '/Users/simone/Desktop/test/ND2conversion/BW_Aldoc.nd2')
    tmp_gene_dir: str 
        path to the folder where to save the npy files
    processing_hyb: str 
        Name of the processing_hyb to process (ex. EXP-17-BP3597_hyb2)
    use_ram: bool 
        If True the images are written first in ram until max_ram is reached
    max_ram: int
        Size of the RAM that can be used as temporary storage
 
    """   
    
    size_counter = 0
    tmp_storage={}
    
    with nd2.Nd2(raw_data_gene_fname) as nd2file:
        for channel in nd2file.channels:
            for fov in nd2file.fields_of_view:
                img_stack = np.empty([len(nd2file.z_levels),nd2file.height,nd2file.width],dtype='uint16')
                images = nd2file.select(channels=channel, fields_of_view=fov,z_levels=nd2file.z_levels)
                for idx,im in enumerate(images):
                    img_stack[idx,:,:] = im
            
                converted_fname=tmp_gene_dir+processing_hyb+'_'+gene+'_pos_'+str(fov)+'.npy'
                
                if use_ram:
                    # Create a dict that is saved in ram. When full is written on disk
                    # Done to reduce the number of i/o and increase performance
                    size_counter += img_stack.nbytes
                    if size_counter < max_ram:
                        tmp_storage[fov]={}
                        tmp_storage[fov]['img']=img_stack
                        tmp_storage[fov]['converted_fname']=converted_fname
                    else:
                        for pos in tmp_storage.keys():
                            np.save(tmp_storage[pos]['converted_fname'],tmp_storage[pos]['img'],allow_pickle=False)

                        tmp_storage={}
                        size_counter = img_stack.nbytes
                        tmp_storage[fov]={}
                        tmp_storage[fov]['img'] = img_stack
                        tmp_storage[fov]['converted_fname'] = converted_fname                  
                
                else:
                    # Directly save the file without saving it in RAM
                    np.save(converted_fname,img_stack,allow_pickle=False)
        
        # Save the data that remained in the tmp_storage (if RAM is used) 
        if use_ram:
            for pos in tmp_storage.keys():
                        np.save(tmp_storage[pos]['converted_fname'],tmp_storage[pos]['img'],allow_pickle=False)