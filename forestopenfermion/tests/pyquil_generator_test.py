"""Testing interface to pyqui.paulis"""
import numpy as np
import pytest
from openfermion.ops import QubitOperator, FermionOperator
from forestopenfermion.pyquil_circuit_generator import exponentiate, TimeEvolution
from pyquil.gates import H, RX, CNOT, RZ
from pyquil.quil import Program


def test_exponentiate():
    one_pauli_term = QubitOperator('X0 Y2 Z3')
    test_program = exponentiate(one_pauli_term)

    true_program = Program().inst([H(0), RX(np.pi/2)(2),
                                   CNOT(0, 2), CNOT(2, 3),
                                   RZ(2.0)(3), CNOT(2, 3),
                                   CNOT(0, 2), H(0),
                                   RX(-np.pi/2)(2)])

    # pyquil has no program compare object
    # string base comparison might fail
    assert true_program.out() == test_program.out()


def test_exponentiate_type_fail():
    fermion_term = FermionOperator('1^ 0')
    with pytest.raises(TypeError):
        exponentiate(fermion_term)


def test_time_evolve():
    one_pauli_term = QubitOperator('X0 Y2 Z3')
    prog = TimeEvolution(1, one_pauli_term)
    true_program = Program().inst([H(0), RX(np.pi/2)(2),
                                   CNOT(0, 2), CNOT(2, 3),
                                   RZ(2.0)(3), CNOT(2, 3),
                                   CNOT(0, 2), H(0),
                                   RX(-np.pi/2)(2)])

    assert isinstance(true_program, Program)
    assert prog.out() == true_program.out()


def test_time_evolve_type_checks():
    one_pauli_term = QubitOperator('X0 Y2 Z3')
    with pytest.raises(TypeError):
        TimeEvolution('a', one_pauli_term)

    with pytest.raises(TypeError):
        TimeEvolution(-1.0*1j, one_pauli_term)

    with pytest.raises(TypeError):
        TimeEvolution(1.0, FermionOperator('1^'))
