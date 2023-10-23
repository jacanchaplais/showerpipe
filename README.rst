showerpipe
==========

|PyPI version| |Tests| |Documentation| |License| |Code style: black|

Provides a Pythonic data pipeline for showering and hadronisation
programs in HEP.

Currently wraps interface for Pythia 8, with plans for Herwig and
Ariadne in future releases.

Installation
------------

Using conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest solution is to just install via conda:

.. code:: bash

   conda install -c jacanchaplais showerpipe

Pythia will be included as a dependency, and all relevant paths and
environment variables will be properly set.

Using existing Pythia installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Pythia 8 is already installed on your system, and available in your
``PYTHONPATH``, you can just install this package via ``pip``.

It also requires the ``PYTHIA8DATA`` environment variable to be set.
This is the path to the ``xmldoc/`` directory under Pythia’s ``share``
directory. You can do something like this in your shell config:

.. code:: bash

   export PYTHIA8DATA=/home/$USER/pythia82xx/share/Pythia8/xmldoc

With everything set up properly, simply run:

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
