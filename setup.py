from setuptools import setup, find_packages

__version__ = "0.0.2"
exec(open('pysmFISH/_version.py').read())

setup(
    name="pysmFISH",
    version=__version__,
    packages=find_packages(),
    python_requires='>3.6',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'matplotlib',
        'nd2reader==2.1.3',
        'sympy',
        'loompy',
        'ruamel.yaml',
        'h5py',
        'dask',
        'distributed',
        'mpi4py'
    ],
    entry_points='''
        [console_scripts]
        add_coords_to_experimental_metadata=add_coords_to_experimental_metadata
        apply_stitching=apply_stitching
        dots_coords_correction=dots_coords_correction
        preprocessing_script=preprocessing_script
        process_standalone_experiment=process_standalone_experiment
        reference_registration=reference_registration
        Run_dots_counting=Run_dots_counting
        staining_segmentation=staining_segmentation
    ''',

    author="Simone Codeluppi",
    author_email="simone.codeluppi@gmail.com",
    keywords=["spatial transcriptomics", "singlecell", "bioinformatics", "transcriptomics"],
    description="Analysis of large spatial transcriptomics data (osmFISH).",
    license="MIT",
    url="https://linnarssonlab.org/osmFISH/",
)