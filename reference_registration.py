import argparse

from pysmFISH import utils
from pysmFISH.stitching_package import hybregistration as hr

def reference_registration():

    """
    This script is used to register the stitched reference channels for the processed
    hybridization. The comparison is sequential (from Hyb1-->HybN) and not all the
    hybridization steps are required.

    The output are pickle files with the recalculated corners according to the 
    registration

    The input parameters are entered via argparse

    Parameters:
    -----------

    path: string. 
        Exact path to the folder with the stitched .sf.hdf5
    reference_gene: string. 
        Reference gene used for stitching
    fraction: float. 
        Fraction of the image to use for the registration. 
        Selection start from the center of the image. 
        Default 0.2

    """

    # Inputs of the function
    parser = argparse.ArgumentParser(description='Register the stitched images \
                                    of the reference channels')

    parser.add_argument('-path', help='path to the folder with the stitched \
                        XX.sf.hdf5 and XX_data_reg.pkl files')
    parser.add_argument('-reference_gene', help='Reference gene used for the \
                        stitching')
    
    parser.add_argument('-fraction',default=0.2, help='fraction of the picture to use for \
                        registration',type=float)

    
    args = parser.parse_args()

    # retrieve the parameters
    processing_directory = args.path
    reference_gene = args.reference_gene
    fraction = args.fraction

    # Determine the operating system running the code
    os_windows, add_slash = utils.determine_os()

    # Check training slash in the processing directory
    processing_directory=utils.check_trailing_slash(processing_directory,os_windows)



    hr.register_final_images_reg_data_only(processing_directory, gene=reference_gene,
                          sub_pic_frac=fraction, use_MPI=False,
                          apply_to_corners=True, apply_warping = False,
                          region=None, compare_in_seq=True)




if __name__ == "__main__":
    reference_registration()