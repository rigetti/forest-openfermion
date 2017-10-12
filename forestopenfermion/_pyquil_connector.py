"""
An interface from OpenFermion QubitObjects to some of the circuit generating functionality in pyquil
"""
from ._pyquil_circuit_generator import qubitop_to_pyquilpauli
from openfermion.ops import QubitOperator

from pyquil.quil import Program
from pyquil.paulis import exponentiate as pyquil_exponentiate


def exponentiate(qubit_operator):
    """
    Generates a pyquil program corresponding to the QubitOperator generator

    The OpenFermion qubit operator is translated to a pyQuil PauliSum which,
    in turn, is passed to the `exponentiate' method. The `exponentiate' method
    generates a circuit that can be simulated with the Forest-qvm or associated
    QVMs.

    Args:
        qubit_operator (QubitOperator):
            Generator of rotations
    Returns:
        Program: a pyQuil program representing the unitary evolution
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

    Converts the Hamiltonian to an instance of the pyQuil Pauliterms and returns
    the time evolution operator. This method mirrors the ProjectQ TimeEvolution
    interface.

    Args:
        time (float, int): time to evolve
        hamiltonian (QubitOperator): a hamiltonian as a OpenFermion
            QubitOperator
    Returns:
        program (Program)
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
