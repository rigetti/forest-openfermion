"""
Test RDM acquisition for accuracy
"""
import os
import numpy as np
from forestopenfermion.rdm_estimation import (pauli_terms_for_tpdm,
                                              pauli_to_tpdm,
                                              pauli_terms_for_tpdm_aa,
                                              pauli_terms_for_tpdm_bb,
                                              pauli_terms_for_tpdm_ab,
                                              pauli_to_tpdm_aa,
                                              pauli_to_tpdm_bb,
                                              pauli_to_tpdm_ab)

from forestopenfermion.rdm_utilities import get_sz_spin_adapted

from forestopenfermion.pyquil_connector import pyquilpauli_to_qubitop
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians import MolecularData
from pyquil.paulis import term_with_coeff


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
    """
    Check if 2-RDM construction from pauli terms works for the aa spin adapted
    block
    """
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    d2aa, d2bb, d2ab = get_sz_spin_adapted(molecule.fci_two_rdm)
    paulis_to_measure = pauli_terms_for_tpdm_aa(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_aa = pauli_to_tpdm_aa(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_aa, d2aa)


def test_lih_tpdm_bb_build():
    """
    Check if 2-RDM construction from pauli terms works for the bb spin adapted
    block
    """
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    d2aa, d2bb, d2ab = get_sz_spin_adapted(molecule.fci_two_rdm)
    paulis_to_measure = pauli_terms_for_tpdm_bb(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_bb = pauli_to_tpdm_bb(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_bb, d2bb)


def test_lih_tpdm_ab_build():
    """
    Check if 2-RDM construction from pauli terms works for the ab block
    """
    lih_file = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=lih_file)
    rdms = molecule.get_molecular_rdm(use_fci=True)
    dim = molecule.n_qubits
    d2aa, d2bb, d2ab = get_sz_spin_adapted(molecule.fci_two_rdm)
    paulis_to_measure = pauli_terms_for_tpdm_ab(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)

    tpdm_ab = pauli_to_tpdm_ab(dim // 2, pauli_to_coeff)
    assert np.allclose(tpdm_ab, d2ab)


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
    d2aa, d2bb, d2ab = get_sz_spin_adapted(tpdm)

    paulis_to_measure_ab = pauli_terms_for_tpdm_ab(dim // 2)
    pauli_to_coeff = {}
    for term in paulis_to_measure_ab:
        # convert back to FermionOperator
        qubit_op = pyquilpauli_to_qubitop(term_with_coeff(term, 1.0))
        pauli_to_coeff[term.id()] = rdms.expectation(qubit_op)
    tpdm_ab = pauli_to_tpdm_ab(dim // 2, pauli_to_coeff)
    assert np.allclose(d2ab, tpdm_ab)
