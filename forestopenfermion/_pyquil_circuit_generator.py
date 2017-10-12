"""
Translates OpenFermion Objects to pyQuil objects
"""
from pyquil.paulis import PauliSum, PauliTerm
from openfermion.ops import QubitOperator


def qubitop_to_pyquilpauli(qubit_operator):
    """
    Convert a OpenFermion QubitOperator to a PauliSum

    Args:
        qubit_operator (QubitOperator): OpenFermion QubitOperator to convert
            to a pyquil PauliSum
    Returns:
        transformed_term (PauliSum)
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError("qubit_operator must be a OpenFermion "
                        "QubitOperator object")

    transformed_term = PauliTerm("I", 0, 0.0)
    for qubit_terms, coefficient in qubit_operator.terms.items():
        base_term = PauliTerm('I', 0)
        for tensor_term in qubit_terms:
            base_term *= PauliTerm(tensor_term[1], tensor_term[0])

        transformed_term += base_term * coefficient

    return transformed_term


def pyquilpauli_to_qubitop(pyquil_pauli):
    """
    Convert a pyQuil PauliSum to a OpenFermion QubitOperator

    Args:
        pyquil_pauli (PauliTerm, PauliSum): pyQuil PauliTerm or PauliSum to
            convert to a OpenFermion QubitOperator
    Returns:
        transformed_term (QubitOperator)
    """
    if not isinstance(pyquil_pauli, (PauliSum, PauliTerm)):
        raise TypeError("pyquil_pauli must be a pyquil PauliSum or "
                        "PauliTerm object")

    if isinstance(pyquil_pauli, PauliTerm):
        pyquil_pauli = PauliSum([pyquil_pauli])

    transformed_term = QubitOperator()
    # iterate through the PauliTerms of PauliSum
    for pauli_term in pyquil_pauli.terms:
        transformed_term += QubitOperator(
            term=tuple(zip(pauli_term._ops.keys(), pauli_term._ops.values())),
            coefficient=pauli_term.coefficient)

    return transformed_term
