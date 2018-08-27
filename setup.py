from setuptools import setup, find_packages

__version__ = "0.1.5"
exec(open('pysmFISH/_version.py').read())

setup(
    name="pysmFISH",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'matplotlib',
        'nd2reader==2.1.3',
        'sympy',
        'ruamel.yaml',
        'h5py',
        'dask',
        'distributed',
        'mpi4py',
        'cython',
        'loompy'
        ],
    scripts=[
        "add_coords_to_experimental_metadata.py",
        "apply_stitching.py",
        "dots_coords_correction.py",
        "preprocessing_script.py",
        "process_standalone_experiment.py",
        "reference_registration.py",
        "Run_dots_counting.py",
        "staining_segmentation.py",
        "run_raw_counting_only.py",
        "run_stitching_reference_only.py"
        ],
    author="Simone Codeluppi",
    author_email="simone.codeluppi@gmail.com",
    keywords=["spatial transcriptomics", "singlecell", "bioinformatics", "transcriptomics"],
    description="Analysis of large spatial transcriptomics data (osmFISH).",
    license="MIT",
    url="https://linnarssonlab.org/osmFISH/",
)
