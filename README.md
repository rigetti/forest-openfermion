The Rigetti Forest Plugin for OpenFermion
=========================================

> **NOTE**: This repository is not being actively developed at Rigetti, and therefore we have
decided to archive it. It should work in its current state, but if you find issues or would like
to suggest that we re-open development on this project (or, even better, if you would like to
develop it!) please send us an email at [software@rigetti.com](mailto:software@rigetti.com).

[![pipeline status](https://gitlab.com/rigetti/forest/forest-openfermion/badges/master/pipeline.svg)](https://gitlab.com/rigetti/forest/forest-openfermion/commits/master)
[![Build Status](https://semaphoreci.com/api/v1/rigetti/forest-openfermion/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/forest-openfermion)

[OpenFermion](http://openfermion.org) is an open-source package for compiling and analyzing
quantum algorithms that simulate fermionic systems. This plugin library allows the circuit
construction and simulation environment [Forest](http://www.rigetti.com/forest) to
interface with OpenFermion.

Getting Started
---------------

To install from source, run the following from inside the top-level directory:

```bash
pip install -e .
```

Alternatively, one can install the last major release from PyPI via:

```bash
pip install forest-openfermion
```

Development and Testing
-----------------------

Tests can be executed from the top-level directory by simply running:

```bash
pytest
```

Note that you will need to have installed the requirements via
`pip install -r requirements.txt` to get `pytest`.
