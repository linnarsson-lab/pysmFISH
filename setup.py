from setuptools import setup, find_packages

__version__ = "0.1.2"
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
    author="Simone Codeluppi",
    author_email="simone.codeluppi@gmail.com",
    keywords=["spatial transcriptomics", "singlecell", "bioinformatics", "transcriptomics"],
    description="Analysis of large spatial transcriptomics data (osmFISH).",
    license="MIT",
    url="https://linnarssonlab.org/osmFISH/",
)
