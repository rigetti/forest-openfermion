"""
Tests for OpenFermion to pyQuil object Translator
"""
import pytest
import numpy as np
from openfermion.ops import (FermionOperator, hermitian_conjugated,
                             QubitOperator, InteractionOperator,
                             InteractionRDM)
from openfermion.transforms import jordan_wigner
from forestopenfermion.pyquil_circuit_generator import (pyquilpauli_to_qubitop,
                                                        qubitop_to_pyquilpauli)
from pyquil.paulis import PauliTerm, PauliSum


def test_confirm_interface():
    """
    Test conversion

    This checks the conversion between pyQuil PauliTerms/Sums
    with OpenFermion QubitOperators. This checks for fields associated with the
    conversion between the two objects. Does not check the field types.
    """
    op = QubitOperator()
    assert hasattr(op, 'terms')
    assert isinstance(op.terms, dict)

    # PauliSum does not allow initialization with an empty value
    op_pyquil = PauliSum([PauliTerm("X", 1)])
    assert hasattr(op_pyquil, 'terms')
    # yell if Forest paulis data structure changes
    assert hasattr(PauliTerm("X", 1), '_ops')


def test_translation_type_enforcement():
    """
    Make sure type check works
    """
    create_one = FermionOperator('1^')
    empty_one_body = np.zeros((2, 2))
    empty_two_body = np.zeros((2, 2, 2, 2))
    interact_one = InteractionOperator(1, empty_one_body, empty_two_body)
    interact_rdm = InteractionRDM(empty_one_body, empty_two_body)

    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(create_one)
    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(interact_one)
    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(interact_rdm)

    # don't accept anything other than pyquil PauliSum or PauliTerm
    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(create_one)
    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(interact_one)
    with pytest.raises(TypeError):
        qubitop_to_pyquilpauli(interact_rdm)


def test_qubitop_to_paulisum():
    """
    Conversion of QubitOperator; accuracy test
    """
    hop_term = FermionOperator(((2, 1), (0, 0)))
    term = hop_term + hermitian_conjugated(hop_term)

    pauli_term = jordan_wigner(term)

    forest_term = qubitop_to_pyquilpauli(pauli_term)
    ground_truth = PauliTerm("X", 0)*PauliTerm("Z", 1)*PauliTerm("X", 2)
    ground_truth += PauliTerm("Y", 0)*PauliTerm("Z", 1)*PauliTerm("Y", 2)
    ground_truth *= 0.5

    assert ground_truth == forest_term


def test_qubitop_to_paulisum_zero():

    identity_term = QubitOperator()
    forest_term = qubitop_to_pyquilpauli(identity_term)
    ground_truth = PauliTerm("I", 0, 0)

    assert ground_truth == forest_term


def test_pyquil_to_qubitop():
    pyquil_term = PauliSum([PauliTerm("X", 0)*PauliTerm("Y", 5)])
    op_fermion_term = QubitOperator(((0, "X"), (5, "Y")))
    test_op_fermion_term = pyquilpauli_to_qubitop(pyquil_term)
    assert test_op_fermion_term.isclose(op_fermion_term)


def test_pyquil_to_qubitop_type_enforced():
    """Enforce the appropriate type"""
    create_one = FermionOperator('1^')
    empty_one_body = np.zeros((2, 2))
    empty_two_body = np.zeros((2, 2, 2, 2))
    interact_one = InteractionOperator(1, empty_one_body, empty_two_body)
    interact_rdm = InteractionRDM(empty_one_body, empty_two_body)

    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(create_one)
    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(interact_one)
    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(interact_rdm)

    # don't accept anything other than pyquil PauliSum or PauliTerm
    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(create_one)
    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(interact_one)
    with pytest.raises(TypeError):
        pyquilpauli_to_qubitop(interact_rdm)


def test_pyquil_to_qubitop_pauliterm_conversion():
    """Test if a pauliterm is converted to a pauli sum"""
    pyquil_term = PauliTerm("X", 0)*PauliTerm("Y", 5)
    # implicit test of conversion from Term to Sum
    open_fermion_term = pyquilpauli_to_qubitop(pyquil_term)
    op_fermion_term = QubitOperator(((0, "X"), (5, "Y")))
    assert open_fermion_term.isclose(op_fermion_term)
