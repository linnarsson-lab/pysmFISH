.. _pipeline_description_steps:

Pipeline steps
==============
 
Image acquisition
-----------------
The images are acquired using a Nikon epifluorescence miscroscope as 3D stacks.
For each channel of each hybridization an .nd2 file with all ``xy`` positions 
is generated.

**File name convention:** ``HybNumber_exp_tag_Gene.nd``


Initial processing
------------------
This step is experiment agnostic. You can preprocess hybridizations imaged for
different experiments together. All the directory inside the top level folder 
are preprocessed (conversion/filtering/raw counting and stitching) independently 
in serial mode however each hyb processing make use of parallel processing.

Create directories
################################

- ``data_to_process``: top level folder that contains all the hybridizations that need to be analyzed.  
- ``exp_tag_hybridization_number``: (Ex: ``EXP-17-BP3597_hyb2``) subfolder containing:
- ``exp_tag_hybridization_number_raw_data`` (Ex: ``EXP-17-BP3597_hyb2_raw_data``) subfolder of the 
  ``exp_tag_hybridization_number`` with the .nd2 files of the genes analyzed in the 
  |current cycle of hybridization  


Create configuration files
##########################

- Copy the template of the :ref:`Experimental_metadata.yaml <configuration>` file in the hyb folder.
- Extract the tiles coords from the raw nikon file and save them as :ref:`HybX_Coords.txt <configuration>`
- Enter the coords into the ``Experimental_metadata.yaml`` file using the :ref:`add_coords_to_experimental_metadata.py <add_coords_exp_metadata>`. 
- Copy the template of the :ref:`Filtering_raw_counting.config.yaml <configuration>` file in the hyb 
  |folder and edit the paramters.
- Copy the template of the :ref:`Staining_segmentation.config.yaml <configuration>` file in the hyb 
  |folder and edit the paramters.


Filtering, raw counting and Stitching
-------------------------------------
Process the hybs saved in a common folder Ex: ``data_to_process`` using the 
:ref:`preprocessing_script.py <preprocessing_script>` The hybs in the processing folder can be from 
different experiments and will be run sequentially.


Consolidate experiment
----------------------
- Transfer all the hyb folders belonging to the same experiment in one single
  folder named with the experiment name (Ex. ``EXP-17-BP3597``). 
  Use the same name that is in the ``Experimental_metadata.yaml`` file. 
- Copy the ``Experimental_metadata.yaml`` that contains the coords of all hybridizations in the experiment folder. 
- Copy the ``Filtering_raw_counting.config.yaml`` of the experiment in the experiment folder.

Transfer the stitched files and data
------------------------------------
- Create the ``stitched_reference_files`` subdirecotry in the experimental directory
- Transfer/copy the ``XX.sf.hdf5`` and ``XX_data.pkl`` into the ``stitched_reference_files``
  subdirectory


Register the stitched images
-------------------------
Register all the stitched images using the :ref:`reference_registration.py<ref_registration>`

Create the stitched images for all genes
----------------------------------------
- This step uses the registered coords.
- Create all the stitched images using :ref:`apply_stitching.py<apply_stitching>`.

Dots coords processing
----------------------
The :ref:`dots_coords_coorection.py<dots_coords_correction>`
script is used to aggregate the raw counting registration_data and to remove 
the overlapping RNA molecules that are counted multiple times in the regions of overlapping between the images.

Staining segmentation
---------------------
The :ref:`staining_segmentation.py<staining_segmentation>` script is used to segment IF or polyA
staining. It returns a dictionary with all the identified objects. The parameters
used for the segmentation are in the ``Staining_segmentation.config.yaml`` file.

Counting dots in segmented regions
----------------------------------
Use the `map_counting.py` to count the RNA molecules in each segmented region.

.. warning::
    the ``map_counting.py`` will be added to the scripts soon. Is currently converted from MPI 
    to dask.distributed.