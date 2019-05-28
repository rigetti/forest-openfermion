The Rigetti Forest Plugin for OpenFermion
=========================================

[![Build Status](https://semaphoreci.com/api/v1/rigetti/forestopenfermion/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/forestopenfermion)

[OpenFermion](http://openfermion.org>) is an open source package for compiling and analyzing
quantum algorithms that simulate fermionic systems. This plugin library allows the circuit
construction and simulation environment [Forest](http://www.rigetti.com/forest>) to
interface with OpenFermion.

Getting Started
---------------

`forestopenfermion` can be installed from source or as a library using `pip`.

To install the source, clone this git repo, change directory to the top level folder and run:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, one can install the last major release with the command

```bash
pip install forestopenfermion
```

Development and Testing
-----------------------

We use `tox` and `pytest` for testing. Tests can be executed from the top-level
directory by simply running:

```bash
tox
```

The setup is currently testing Python 2.7 and Python 3.6.
