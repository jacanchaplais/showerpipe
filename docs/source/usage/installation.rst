============
Installation
============

The easiest way to install showerpipe is to download the environment.yml file
in the root of this project, and create a virtual environment from it using
conda. This sets up the Pythia Python interface automatically, including the
fiddly aspects of the necessary environment variables. It also includes
OpenMP's MPI and libraries compatible with it. Use the command::

    conda env create -f environment.yml

This will create a virtual environment called showerpipe, which can be
activated using::

    conda activate showerpipe

Alternatively, you can use your own Pythia installation, which does include
a Python interface, but you will need to manually set the ``PYTHONPATH`` and
``PYTHIA8DATA`` environment variables. If you wish to use the parallel
functionality of showerpipe, you will also need to install mpi4py, and
compatible versions of hdf5 and h5py.
