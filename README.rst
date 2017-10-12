Fores-OpenFermion
==================


`OpenFermion <http://openfermion.org>`_ is an open source package for compiling and analyzing quantum algorithms that simulate fermionic systems.
This plugin library allows the circuit construction and simulation enviornment `Forest <http://www.rigetti.com/forest>`_ to interface with OpenFermion.

Getting started
---------------

To start using Forest-OpenFermion, first install `OpenFermion <http://openfermion.org>`_ and the libraries associated
with Forest `pyQuil <https://github.com/rigetticomputing/pyQuil>`_, and `grove <https://github.com/rigetticomputing/grove>`_.
Then, clone this git repo, change directory to the top level folder and run:

.. code-block:: bash

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


