############################################################################
#   Copyright 2017 Rigetti Computing, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
############################################################################
"""
An interface from OpenFermion QubitObjects to some of the circuit generating functionality in pyquil
"""
from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli
from openfermion.ops import QubitOperator

from pyquil.quil import Program
from pyquil.paulis import exponentiate as pyquil_exponentiate


def exponentiate(qubit_operator):
    """
    Generates a pyquil program corresponding to the QubitOperator generator.

    The OpenFermion qubit operator is translated to a pyQuil PauliSum which, in turn, is passed to
    the `exponentiate' method. The `exponentiate' method generates a circuit that can be simulated
    with the Forest-qvm or associated QVMs.

    :param QubitOperator qubit_operator: Generator of rotations
    :return: a pyQuil program representing the unitary evolution
    :rtype: Program
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError("qubit_operator must be an OpenFermion "
                        "QubitOperator type")

    pauli_sum_representation = qubitop_to_pyquilpauli(qubit_operator)
    prog = Program()
    for term in pauli_sum_representation.terms:
        prog += pyquil_exponentiate(term)

    return prog


def TimeEvolution(time, hamiltonian):
    """
    Time evolve a hamiltonian

    Converts the Hamiltonian to an instance of the pyQuil Pauliterms and returns the time evolution
    operator. This method mirrors the ProjectQ TimeEvolution interface.

    :param [float, int] time: time to evolve
    :param QubitOperator hamiltonian: a Hamiltonian as a OpenFermion QubitOperator
    :return: a pyquil Program representing the Hamiltonian
    :rtype: Program
    """
    if not isinstance(time, (int, float)):
        raise TypeError("float must be a float or an int")
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError("hamiltonian must be an OpenFermion "
                        "QubitOperator object")

    pyquil_pauli_term = qubitop_to_pyquilpauli(hamiltonian)
    pyquil_pauli_term *= time
    prog = Program()
    for term in pyquil_pauli_term.terms:
        prog += pyquil_exponentiate(term)

    return prog
