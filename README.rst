Fores-OpenFermion
==================


`OpenFermion <http://openfermion.org>`_ is an open source package for compiling and analyzing quantum algorithms that simulate fermionic systems.
This plugin library allows the circuit construction and simulation enviornment `Forest <http://www.rigetti.com/forest>`_ to interface with OpenFermion.

Getting started
---------------

Forest-OpenFermion can be installed from source or as a library using pip.

To install the source, clone this git repo, change directory to the top level folder and run:

.. code-block:: bash

  pip install -r requirements.txt
  python -m pip install -e .

Alternatively, one can install the last major release with the command

.. code-block:: bash

  python -m pip install forestopenfermion


Development and Testing
-----------------------

We use tox and pytest for testing. Tests can be executed from the top-level directory by simply running:

.. code-block:: bash
  tox

The setup is currently testing Python 2.7 and Python 3.6.
