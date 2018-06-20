"""
Test RDM acquisition for accuracy
"""
import sys
import os
import numpy as np
from itertools import product
from forestopenfermion.rdm_estimation import (pauli_terms_for_tpdm,
                                              pauli_to_tpdm,
                                              pauli_terms_for_tpdm_aa,
                                              pauli_terms_for_tpdm_bb,
                                              pauli_terms_for_tpdm_ab,
                                              pauli_to_tpdm_aa,
                                              pauli_to_tpdm_bb,
                                              pauli_to_tpdm_ab,
                                              pauli_term_relabel)

from forestopenfermion.pyquil_connector import pyquilpauli_to_qubitop
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians import MolecularData
from pyquil.paulis import term_with_coeff, PauliTerm


def get_sz_spin_adapted(measured_tpdm):
    """
    Take a spin-orbital 4-tensor 2-RDM and map to the SZ spin adapted version return aa, bb, and ab matrices

    :param measured_tpdm: spin-orbital 2-RDM 4-tensor
    :return: 2-RDM matrices for aa, bb, and ab
    """
    if np.ndim(measured_tpdm) != 4:
        raise TypeError("measured_tpdm must be a 4-tensor")

    dim = measured_tpdm.shape[0]  # spin-orbital basis rank
    # antisymmetric basis dimension
    aa_dim = int((dim / 2) * (dim / 2 - 1) / 2)
    ab_dim = int((dim / 2) ** 2)
    d2_aa = np.zeros((aa_dim, aa_dim))
    d2_bb = np.zeros_like(d2_aa)
    d2_ab = np.zeros((ab_dim, ab_dim))

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    # iterate over spatial orbital indices
    for p, q in product(range(dim // 2), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    for i, j, k, l in product(range(dim // 2), repeat=4):  # iterate over spatial indices
        d2_ab[bas_ab[(i, j)], bas_ab[(k, l)]] = measured_tpdm[2 * i, 2 * j + 1, 2 * k, 2 * l + 1].real

        if i < j and k < l:
            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[2 * i, 2 * j, 2 * k, 2 * l].real - \
                                                    measured_tpdm[2 * i, 2 * j, 2 * l, 2 * k].real - \
                                                    measured_tpdm[2 * j, 2 * i, 2 * k, 2 * l].real + \
                                                    measured_tpdm[2 * j, 2 * i, 2 * l, 2 * k].real

            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1].real - \
                                                    measured_tpdm[2 * i + 1, 2 * j + 1, 2 * l + 1, 2 * k + 1].real - \
                                                    measured_tpdm[2 * j + 1, 2 * i + 1, 2 * k + 1, 2 * l + 1].real + \
                                                    measured_tpdm[2 * j + 1, 2 * i + 1, 2 * l + 1, 2 * k + 1].real

            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5

    return d2_aa, d2_bb, d2_ab


def test_relabel_terms():
    """
    Check if Pauli term relabeling works
    """
    n_qubits = 4
    test_operator =  PauliTerm.from_list([('Y', 0), ('X', 1), ('Y', 2), ('X', 3)])
    test_operator += PauliTerm.from_list([('Y', 0), ('X', 1), ('X', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list([('Y', 0), ('Y', 1), ('Y', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list([('Y', 0), ('Y', 1), ('X', 2), ('X', 3)])
    test_operator += PauliTerm.from_list([('X', 0), ('X', 1), ('Y', 2), ('Y', 3)])
    test_operator += PauliTerm.from_list([('X', 0), ('X', 1), ('X', 2), ('X', 3)])
    test_operator += PauliTerm.from_list([('X', 0), ('Y', 1), ('Y', 2), ('X', 3)])
    test_operator += PauliTerm.from_list([('X', 0), ('Y', 1), ('X', 2), ('Y', 3)])

    true_operator =  PauliTerm.from_list([('Y', 4), ('X', 5), ('Y', 6), ('X', 7)])
    true_operator += PauliTerm.from_list([('Y', 4), ('X', 5), ('X', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list([('Y', 4), ('Y', 5), ('Y', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list([('Y', 4), ('Y', 5), ('X', 6), ('X', 7)])
    true_operator += PauliTerm.from_list([('X', 4), ('X', 5), ('Y', 6), ('Y', 7)])
    true_operator += PauliTerm.from_list([('X', 4), ('X', 5), ('X', 6), ('X', 7)])
    true_operator += PauliTerm.from_list([('X', 4), ('Y', 5), ('Y', 6), ('X', 7)])
    true_operator += PauliTerm.from_list([('X', 4), ('Y', 5), ('X', 6), ('Y', 7)])

    label_map = dict(zip(range(n_qubits), range(n_qubits, 2 * n_qubits + 1)))
    relabeled_terms = pauli_term_relabel(test_operator.terms, label_map)
    assert sum(relabeled_terms) == true_operator


def test_h2_tpdm_build():
    """
    Check if constructing the 2-RDM (full) works.  This check uses openfermion
    data and thus requires openfermion to be installed
    """

    h2_file = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7.hdf5")
    molecule = MolecularData(filename=h2_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    paulis_to_measure = pauli_terms_for_tpdm(dim)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm = pauli_to_tpdm(dim, pauli_to_coeff)
    assert np.allclose(tpdm, molecule.fci_two_rdm)


def test_lih_tpdm_aa_build():
    print("lih aa")
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = np.einsum('pqrs->pqsr', molecule.fci_two_rdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)
    paulis_to_measure = pauli_terms_for_tpdm_aa(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_aa = pauli_to_tpdm_aa(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_aa, d2aa)


def test_lih_tpdm_bb_build():
    print("lih bb")
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = np.einsum('pqrs->pqsr', molecule.fci_two_rdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)
    paulis_to_measure = pauli_terms_for_tpdm_bb(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_bb = pauli_to_tpdm_bb(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_bb, d2bb)


def test_lih_tpdm_ab_build():
    print("lih ab")
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = np.einsum('pqrs->pqsr', molecule.fci_two_rdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)
    paulis_to_measure = pauli_terms_for_tpdm_ab(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_ab = pauli_to_tpdm_ab(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_ab, d2ab)


def test_lih_tpdm_build():
    print("lih full")
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    paulis_to_measure = pauli_terms_for_tpdm(dim)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm = pauli_to_tpdm(dim, pauli_to_coeff)
    assert np.allclose(tpdm, molecule.fci_two_rdm)


def test_h2_spin_adapted_aa():
    h2_file = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7.hdf5")
    molecule = MolecularData(filename=h2_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    paulis_to_measure = pauli_terms_for_tpdm(dim)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm = pauli_to_tpdm(dim, pauli_to_coeff)
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_aa = pauli_terms_for_tpdm_aa(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_aa:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_aa = pauli_to_tpdm_aa(dim // 2, pauli_to_coeff)

    assert np.allclose(d2aa, tpdm_aa)


def test_h2_spin_adapted_bb():
    h2_file = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7.hdf5")
    molecule = MolecularData(filename=h2_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    paulis_to_measure = pauli_terms_for_tpdm(dim)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm = pauli_to_tpdm(dim, pauli_to_coeff)
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_bb = pauli_terms_for_tpdm_bb(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_bb:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_bb = pauli_to_tpdm_bb(dim // 2, pauli_to_coeff)

    assert np.allclose(d2bb, tpdm_bb)


def test_h2_spin_adapted_ab():
    h2_file = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7.hdf5")
    molecule = MolecularData(filename=h2_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    paulis_to_measure = pauli_terms_for_tpdm(dim)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm = pauli_to_tpdm(dim, pauli_to_coeff)
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_ab = pauli_terms_for_tpdm_ab(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_ab:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_ab = pauli_to_tpdm_ab(dim // 2, pauli_to_coeff)
    assert np.allclose(d2ab, tpdm_ab)



def test_lih_spin_adapted_aa():
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = molecule.fci_two_rdm
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_aa = pauli_terms_for_tpdm_aa(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_aa:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_aa = pauli_to_tpdm_aa(dim // 2, pauli_to_coeff)

    assert np.allclose(d2aa, tpdm_aa)


def test_lih_spin_adapted_bb():
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = molecule.fci_two_rdm
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_bb = pauli_terms_for_tpdm_bb(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_bb:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_bb = pauli_to_tpdm_bb(dim // 2, pauli_to_coeff)

    assert np.allclose(d2bb, tpdm_bb)


def test_lih_spin_adapted_ab():
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    tpdm = molecule.fci_two_rdm
    tpdm = np.einsum('pqrs->pqsr', tpdm)
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_ab = pauli_terms_for_tpdm_ab(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_ab:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_ab = pauli_to_tpdm_ab(dim // 2, pauli_to_coeff)
    assert np.allclose(d2ab, tpdm_ab)


if __name__ == "__main__":
    test_lih_tpdm_aa_build()
    test_lih_tpdm_bb_build()
    test_lih_tpdm_ab_build()
    test_lih_tpdm_build()

    test_h2_spin_adapted_aa()
    test_h2_spin_adapted_bb()
    test_h2_spin_adapted_ab()

    test_lih_spin_adapted_aa()
    test_lih_spin_adapted_bb()
    test_lih_spin_adapted_ab()

