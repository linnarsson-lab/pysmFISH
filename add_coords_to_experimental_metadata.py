from pysmFISH import utils
import argparse

def add_coords_to_experimental_metadata():

    """
    Script used to add the coords to the Experimental_metadata.yaml file.
    In order to work: 

    - The Experimental_metadata.yaml file needs to have the processing hybridization 
    and the coords at point 0 initialized: ex:  

    TilesPositions:  
    Hybridization1:  
        0:  

    - The coords file extracted from the Nikon microscope need to be in a text file
    ex. Hyb1_Coords.txt. The Coords tag is necessary and Hyb is the hybridization_key

    - The input parameters are passed via argparse

    Parameters:
    -----------
    hybridization_number: string
        Hybridization number (ex. 3) describing the hybridization to process.
    
    path: string 
        Exact path to the folder containing the text file with the coordinates.
    
    hybridization_key: string 
        Possible values 'Hyb' or 'Strip'. To add coordinates for stripping if necessary.

    """



    # Inputs of the function
    parser = argparse.ArgumentParser(description='Add microscope coords to \
                                    experimental_metadata.yaml file')
    parser.add_argument('-hybridization_number', help='hybridization to add the coords to')
    parser.add_argument('-path', help='path to the folder with the experimental_metadata.yaml file')
    parser.add_argument('-hybridization_key', default='Hyb',help='String to identify the specific \
                        coords file')
    args = parser.parse_args()

    # retrieve the parameters
    processing_directory = args.path
    hyb_nr = args.hybridization_number
    hyb_key = args.hybridization_key

    # Determine the operating system running the code
    os_windows, add_slash = utils.determine_os()

    # Check training slash in the processing directory
    processing_directory=utils.check_trailing_slash(processing_directory,os_windows)

    utils.add_coords_to_yaml(processing_directory,hyb_nr,hyb_key)



if __name__ == "__main__":
    add_coords_to_experimental_metadata()


