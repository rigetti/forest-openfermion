"""
Testing rdm spin adapting and pauli term label mapping
"""
import os
import numpy as np
from forestopenfermion.rdm_utilities import (get_sz_spin_adapted, unspin_adapt,
                                             pauli_term_from_string,
                                             pauli_dict_relabel,
                                             pauli_term_relabel)
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians import MolecularData
from pyquil.paulis import PauliTerm


def test_spin_adapt_h2():
    """
    Test if we accurately generate the fci spin-adapted d2-matrices
    """
    h2_file = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7.hdf5")
    molecule = MolecularData(filename=h2_file)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(molecule.fci_two_rdm)
    assert np.allclose(d2aa, np.zeros((1, 1)))
    assert np.allclose(d2bb, np.zeros((1, 1)))

    true_d2ab = np.array([[0.98904311, 0., 0., -0.10410015],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [-0.10410015, 0., 0., 0.01095689]])
    assert np.allclose(d2ab, true_d2ab)

    test_tpdm = unspin_adapt(d2aa, d2bb, d2ab)
    assert np.allclose(molecule.fci_two_rdm, test_tpdm)


def test_spin_adapt_lih():
    """
    Test if we accurately generate the fci spin-adapted d2-matrices
    """
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(molecule.fci_two_rdm)
    assert np.allclose(d2aa, d2bb)
    assert np.isclose(np.trace(d2ab), (molecule.n_electrons / 2)**2)

    test_tpdm = unspin_adapt(d2aa, d2bb, d2ab)
    assert np.allclose(molecule.fci_two_rdm, test_tpdm)


def test_relabel_terms():
    """
    Check if Pauli term relabeling works
    """
    n_qubits = 4
    test_operator = PauliTerm.from_list(
        [('Y', 0), ('X', 1), ('Y', 2), ('X', 3)])
    test_operator += PauliTerm.from_list(
        [('Y', 0), ('X', 1), ('X', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list(
        [('Y', 0), ('Y', 1), ('Y', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list(
        [('Y', 0), ('Y', 1), ('X', 2), ('X', 3)])
    test_operator += PauliTerm.from_list(
        [('X', 0), ('X', 1), ('Y', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list(
        [('X', 0), ('X', 1), ('X', 2), ('X', 3)])
    test_operator += PauliTerm.from_list(
        [('X', 0), ('Y', 1), ('Y', 2), ('X', 3)])
    test_operator += PauliTerm.from_list(
        [('X', 0), ('Y', 1), ('X', 2), ('Y', 3)])

    true_operator = PauliTerm.from_list(
        [('Y', 4), ('X', 5), ('Y', 6), ('X', 7)])
    true_operator += PauliTerm.from_list(
        [('Y', 4), ('X', 5), ('X', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list(
        [('Y', 4), ('Y', 5), ('Y', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list(
        [('Y', 4), ('Y', 5), ('X', 6), ('X', 7)])
    true_operator += PauliTerm.from_list(
        [('X', 4), ('X', 5), ('Y', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list(
        [('X', 4), ('X', 5), ('X', 6), ('X', 7)])
    true_operator += PauliTerm.from_list(
        [('X', 4), ('Y', 5), ('Y', 6), ('X', 7)])
    true_operator += PauliTerm.from_list(
        [('X', 4), ('Y', 5), ('X', 6), ('Y', 7)])

    label_map = dict(
        zip(range(n_qubits), range(n_qubits, 2 * n_qubits + 1)))
    relabeled_terms = pauli_term_relabel(test_operator.terms, label_map)
    assert sum(relabeled_terms) == true_operator


def test_relabeled_result_dictionary():
    result_dictionary = {'': 1.0, 'Z0': (-0.97419999999999995-0j),
                         'Z2': (0.97419999999999995-0j), 'Z0Z2': (-1+0j)}
    relabled_result_dictionary = {'': 1.0, 'Z2': (-0.97419999999999995-0j),
                                  'Z3': (0.97419999999999995-0j), 'Z2Z3': (-1+0j)}

    label_map = {0: 2, 2: 3}
    test_relabeled = pauli_dict_relabel(result_dictionary,
                                        label_map)

    assert relabled_result_dictionary == test_relabeled


def test_term_from_string():
    pauli_string = 'X0Z1Y3I4'
    true_pauli_term = PauliTerm.from_list([('X', 0), ('Z', 1), ('Y', 3)])
    test_pauli_term = pauli_term_from_string(pauli_string)
    assert test_pauli_term == true_pauli_term
