The Rigetti Forest Plugin for OpenFermion
=========================================

[![Build Status](https://semaphoreci.com/api/v1/rigetti/forestopenfermion/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/forestopenfermion)

[OpenFermion](http://openfermion.org>) is an open source package for compiling and analyzing
quantum algorithms that simulate fermionic systems. This plugin library allows the circuit
construction and simulation environment [Forest](http://www.rigetti.com/forest>) to
interface with OpenFermion.

Getting Started
---------------

To install from source, run the following from inside the top-level directory:

```bash
pip install -e .
```

Alternatively, one can install the last major release from PyPI via:

```bash
pip install forestopenfermion
```

Development and Testing
-----------------------

We use `pytest` for testing. Tests can be executed from the top-level
directory by simply running:

```bash
pytest
```

Note that you will need to have installed the requirements via
`pip install -r requirements.txt` to get `pytest`.
