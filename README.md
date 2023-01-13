# showerpipe

[![PyPI version](https://img.shields.io/pypi/v/showerpipe.svg)](https://pypi.org/project/showerpipe/)
[![Documentation](https://readthedocs.org/projects/showerpipe/badge/?version=latest)](https://showerpipe.readthedocs.io)
![Tests](https://github.com/jacanchaplais/showerpipe/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Provides a Pythonic data pipeline for showering and hadronisation programs in
HEP.

Currently wraps interface for Pythia 8, with plans for Herwig and Ariadne in
future releases.

## Installation

This package requires Pythia 8 to be installed on your system, and available in
your `PYTHONPATH`.

It also requires the `PYTHIA8DATA` environment variable to be set. This is the
path to the `xmldoc/` directory under Pythia's `share` directory.
You can do something like this in your shell config:
```bash
export PYTHIA8DATA=/home/$USER/pythia82xx/share/Pythia8/xmldoc
```

### Without an existing Pythia installation (using conda)

If this is not already the case, a very convenient solution is to install it
via `conda`. This is fast, and automatically sets the `PYTHIA8DATA` environment
variable when you activate the virtual environment. An environment file is
provided in the root of this repo, which will install all requirements and then
showerpipe, automatically. The virtual environment can be created using:
```bash
conda env create -f environment.yml
```

If you have an existing conda environment, you can update it by activating the
environment and then using:
```bash
conda env update -f environment.yml --prune
```

### With existing Pythia installation

Simply:
```bash
pip install showerpipe
```
