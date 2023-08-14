showerpipe
==========

|PyPI version| |Tests| |Documentation| |License| |Code style: black|

Provides a Pythonic data pipeline for showering and hadronisation
programs in HEP.

Currently wraps interface for Pythia 8, with plans for Herwig and
Ariadne in future releases.

Installation
------------

This package requires Pythia 8 to be installed on your system, and
available in your ``PYTHONPATH``.

It also requires the ``PYTHIA8DATA`` environment variable to be set.
This is the path to the ``xmldoc/`` directory under Pythia’s ``share``
directory. You can do something like this in your shell config:

.. code:: bash

   export PYTHIA8DATA=/home/$USER/pythia82xx/share/Pythia8/xmldoc

Without an existing Pythia installation (using conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If this is not already the case, a very convenient solution is to
install it via ``conda``. This is fast, and automatically sets the
``PYTHIA8DATA`` environment variable when you activate the virtual
environment. An environment file is provided in the root of this repo,
which will install all requirements and then showerpipe, automatically.
The virtual environment can be created using:

.. code:: bash

   conda env create -f environment.yml

If you have an existing conda environment, you can update it by
activating the environment and then using:

.. code:: bash

   conda env update -f environment.yml --prune

With existing Pythia installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply:

.. code:: bash

   pip install showerpipe

.. |PyPI version| image:: https://img.shields.io/pypi/v/showerpipe.svg
   :target: https://pypi.org/project/showerpipe/
.. |Tests| image:: https://github.com/jacanchaplais/showerpipe/actions/workflows/tests.yml/badge.svg
.. |Documentation| image:: https://readthedocs.org/projects/showerpipe/badge/?version=latest
   :target: https://showerpipe.readthedocs.io
.. |License| image:: https://img.shields.io/pypi/l/showerpipe
   :target: https://raw.githubusercontent.com/jacanchaplais/showerpipe/main/LICENSE.txt
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
