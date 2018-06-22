"""
A module for generating pauli terms corresponding to the 2-RDMs
"""
import numpy as np
from itertools import product
from grove.measurements.estimation import (remove_imaginary_terms,
                                           estimate_pauli_sum,
                                           remove_identity)
from grove.measurements.term_grouping import commuting_sets_by_zbasis
from pyquil.paulis import PauliTerm, PauliSum

from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from forestopenfermion.rdm_utilities import (pauli_dict_relabel,
                                             pauli_term_relabel)
from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli


def _measure_list_of_pauli_terms(pauli_terms, variance_bound, program,
                                 quantum_resource):
    """
    Measure the expected value of a list of Pauli terms and return as a dict

    :param pauli_terms: Pauli Terms to measure
    :param variance_bound: variance bound for measurement.  Right now this is
                           the bound on the variance if you summed up all the
                           individual terms. 1.0E-6 is a good place to start.
    :param program: pyquil Program preparing state
    :param quantum_resource: quantum abstract machine connection object
    :return: results dictionary where the key is the Pauli term ID and the value
             is the expected value
    """
    # group them into commuting sets and then measure
    grouped_terms = commuting_sets_by_zbasis(sum(pauli_terms))

    # measure the terms
    result_dictionary = {}
    for key, terms in grouped_terms.items():
        pauli_sum, identity_term = remove_identity(terms)
        if isinstance(identity_term, int):
            # no identity term
            pass
        elif isinstance(identity_term, PauliSum):
            result_dictionary[identity_term[0].id()] = 1.0
        else:
            print(identity_term, type(identity_term))
            raise TypeError("This type is not recognized for identity_term")

        results = estimate_pauli_sum(pauli_sum, dict(key), program,
                                     variance_bound / len(terms),
                                     quantum_resource)
        for idx, term in enumerate(pauli_sum.terms):
            result_dictionary[term.id()] = results.pauli_expectations[idx] / \
                                           term.coefficient
    return result_dictionary


def measure_aa_tpdm(spatial_dim, variance_bound, program, quantum_resource,
                    transform=jordan_wigner, label_map=None):
    """
    Measure the alpha-alpha block of the 2-RDM

    :param spatial_dim: size of spatial basis function
    :param variance_bound: variance bound for measurement.  Right now this is
                           the bound on the variance if you summed up all the
                           individual terms. 1.0E-6 is a good place to start.
    :param program: a pyquil Program
    :param quantum_resource: a quantum abstract machine connection object
    :param transform: fermion-to-qubit transform
    :param label_map: qubit label re-mapping if different physical qubits are
                      desired
    :return: the alpha-alpha block of the 2-RDM
    """
    # first get the pauli terms corresponding to the alpha-alpha block
    pauli_terms_in_aa = pauli_terms_for_tpdm_aa(spatial_dim,
                                                transform=jordan_wigner)
    if label_map is not None:
        pauli_terms_in_aa = pauli_term_relabel(sum(pauli_terms_in_aa),
                                               label_map)
        rev_label_map = dict(zip(label_map.values(), label_map.keys()))

    result_dictionary = _measure_list_of_pauli_terms(pauli_terms_in_aa,
                                                     variance_bound,
                                                     program,
                                                     quantum_resource)
    if label_map is not None:
        result_dictionary = pauli_dict_relabel(result_dictionary, rev_label_map)

    d2aa = pauli_to_tpdm_aa(spatial_dim, result_dictionary, transform=transform)
    return d2aa


def measure_bb_tpdm(spatial_dim, variance_bound, program, quantum_resource,
                    transform=jordan_wigner, label_map=None):
    """
    Measure the beta-beta block of the 2-RDM

    :param spatial_dim: size of spatial basis function
    :param variance_bound: variance bound for measurement.  Right now this is
                           the bound on the variance if you summed up all the
                           individual terms. 1.0E-6 is a good place to start.
    :param program: a pyquil Program
    :param quantum_resource: a quantum abstract machine connection object
    :param transform: fermion-to-qubit transform
    :param label_map: qubit label re-mapping if different physical qubits are
                      desired
    :return: the beta-beta block of the 2-RDM
    """
    # first get the pauli terms corresponding to the alpha-alpha block
    pauli_terms_in_bb = pauli_terms_for_tpdm_bb(spatial_dim,
                                                transform=jordan_wigner)
    if label_map is not None:
        pauli_terms_in_bb = pauli_term_relabel(sum(pauli_terms_in_bb),
                                               label_map)
        rev_label_map = dict(zip(label_map.values(), label_map.keys()))

    result_dictionary = _measure_list_of_pauli_terms(pauli_terms_in_bb,
                                                     variance_bound,
                                                     program,
                                                     quantum_resource)
    if label_map is not None:
        result_dictionary = pauli_dict_relabel(result_dictionary, rev_label_map)

    d2bb = pauli_to_tpdm_bb(spatial_dim, result_dictionary, transform=transform)
    return d2bb


def measure_ab_tpdm(spatial_dim, variance_bound, program, quantum_resource,
                    transform=jordan_wigner, label_map=None):
    """
    Measure the alpha-beta block of the 2-RDM

    :param spatial_dim: size of spatial basis function
    :param variance_bound: variance bound for measurement.  Right now this is
                           the bound on the variance if you summed up all the
                           individual terms. 1.0E-6 is a good place to start.
    :param program: a pyquil Program
    :param quantum_resource: a quantum abstract machine connection object
    :param transform: fermion-to-qubit transform
    :param label_map: qubit label re-mapping if different physical qubits are
                      desired
    :return: the alpha-beta block of the 2-RDM
    """
    # first get the pauli terms corresponding to the alpha-alpha block
    pauli_terms_in_ab = pauli_terms_for_tpdm_ab(spatial_dim,
                                                transform=jordan_wigner)
    if label_map is not None:
        pauli_terms_in_ab = pauli_term_relabel(sum(pauli_terms_in_ab),
                                               label_map)
        rev_label_map = dict(zip(label_map.values(), label_map.keys()))

    result_dictionary = _measure_list_of_pauli_terms(pauli_terms_in_ab,
                                                     variance_bound,
                                                     program,
                                                     quantum_resource)

    if label_map is not None:
        result_dictionary = pauli_dict_relabel(result_dictionary, rev_label_map)

    d2ab = pauli_to_tpdm_ab(spatial_dim, result_dictionary, transform=transform)
    return d2ab


def pauli_terms_for_tpdm_ab(spatial_dim, transform=jordan_wigner):
    """
    Build the alpha-beta block of the 2-RDM

    Note: OpenFermion ordering is used.  The 2-RDM(alpha-beta) block
          is spatial_dim**2 linear size. (DOI: 10.1021/acs.jctc.6b00190)

    :param spatial_dim: rank of spatial orbitals in the basis set.
    :param transform: type of fermion-to-qubit transformation.
    :return: list of Pauli terms to measure required to construct a the alpha-
             beta block.
    """
    # build basis look up table
    bas_ab = {}
    cnt_ab = 0
    # iterate over spatial orbital indices
    for p, q in product(range(spatial_dim), repeat=2):
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    pauli_terms_to_measure = []
    pauli_to_rdm = {}
    for p, q, r, s in product(range(spatial_dim), repeat=4):
        spin_adapted_term = FermionOperator(
            ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)))

        tpdm_element_as_pauli = remove_imaginary_terms(
            qubitop_to_pyquilpauli(transform(spin_adapted_term)))
        for term in tpdm_element_as_pauli:
            pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()),
                                   key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(
            map(lambda x: (x[1], x[0]), pauli_tensor_list))
        pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                         coefficient=term.coefficient)

        if pauli_term.id() not in pauli_to_rdm.keys():
            pauli_to_rdm[pauli_term.id()] = pauli_term
        else:
            if (abs(pauli_to_rdm[pauli_term.id()].coefficient) <
                    abs(pauli_term.coefficient)):
                pauli_to_rdm[pauli_term.id()] = pauli_term

    return list(pauli_to_rdm.values())


def pauli_to_tpdm_ab(spatial_dim, pauli_to_coeff, transform=jordan_wigner):
    """
    Populate the alpha-beta block of the 2-RDM

    Given a dictionary of expected values of Pauli terms, populate the
    alpha-beta block of the 2-RDM

    :param Int spatial_dim: spatial basis set rank
    :param pauli_to_coeff: a map between the Pauli term label to the expected
                           value.
    :param transform: Openfermion fermion-to-qubit transform
    :return: the 2-RDM alpha-beta block of the 2-RDM
    """
    d2_ab = np.zeros((spatial_dim ** 2, spatial_dim ** 2), dtype=complex)
    # build basis look up table
    bas_ab = {}
    cnt_ab = 0
    # iterate over spatial orbital indices
    for p, q in product(range(spatial_dim), repeat=2):
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    for p, q, r, s in product(range(spatial_dim), repeat=4):
        spin_adapted_term = FermionOperator(
            ((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)))

        tpdm_element_as_pauli = remove_imaginary_terms(
            qubitop_to_pyquilpauli(transform(spin_adapted_term)))

        for term in tpdm_element_as_pauli:
            pauli_tensor_list = sorted(list(
                term.operations_as_set()), key=lambda x: x[0])
            rev_order_pauli_tensor_list = list(
                map(lambda x: (x[1], x[0]), pauli_tensor_list))
            pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                             coefficient=term.coefficient)
            try:
                d2_ab[bas_ab[(p, q)], bas_ab[(s, r)]] += pauli_to_coeff[
                                                         pauli_term.id()] * \
                                                         pauli_term.coefficient
            except KeyError:
                raise Warning("key was not in the coeff matrix.")
    return d2_ab


def pauli_terms_for_tpdm_aa(spatial_dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the alpha-alpha
    block of the 2-RDM

    Given a dictionary of expected values of Pauli terms, populate the
    alpha-beta block of the 2-RDM

    :param Int spatial_dim: Dimension of the spatial-orbital basis.
    :param transform: fermion-to-qubit transform from OpenFermion
    :return: List of PauliTerms that corresponds to set of pauli terms to
             measure to construct the 2-RDM.
    """
    # build basis lookup table
    bas_aa = {}
    cnt_aa = 0
    for p, q in product(range(spatial_dim), repeat=2):
        if q < p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1

    pauli_terms_to_measure = []
    pauli_to_rdm = {}
    for p, q, r, s in product(range(spatial_dim), repeat=4):
        if p < q and s < r:
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}
            # a_{q, \alpha}^{\dagger} -
            # a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(
                ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))) - \
                                FermionOperator(((2 * p, 1), (2 * q, 1),
                                                 (2 * s, 0), (2 * r, 0))) - \
                                FermionOperator(((2 * q, 1), (2 * p, 1),
                                                 (2 * r, 0), (2 * s, 0))) + \
                                FermionOperator(((2 * q, 1), (2 * p, 1),
                                                 (2 * s, 0), (2 * r, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))
            for term in tpdm_element_as_pauli:
                pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()),
                                   key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(
            map(lambda x: (x[1], x[0]), pauli_tensor_list))
        pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                         coefficient=term.coefficient)

        if pauli_term.id() not in pauli_to_rdm.keys():
            pauli_to_rdm[pauli_term.id()] = pauli_term
        else:
            if (abs(pauli_to_rdm[pauli_term.id()].coefficient) <
                    abs(pauli_term.coefficient)):
                pauli_to_rdm[pauli_term.id()] = pauli_term

    return list(pauli_to_rdm.values())


def pauli_to_tpdm_aa(spatial_dim, pauli_to_coeff, transform=jordan_wigner):
    """
    Populate the alpha-alpha block of the 2-RDM

    :param Int spatial_dim: spatial basis set rank
    :param pauli_to_coeff: a map between the Pauli term label to the expected
                           value.
    :param transform: Openfermion fermion-to-qubit transform
    :return: the 2-RDM alpha-beta block of the 2-RDM
    """
    aa_dim = int(spatial_dim * (spatial_dim - 1) / 2)
    d2_aa = np.zeros((aa_dim, aa_dim), dtype=complex)

    # build basis lookup table
    bas_aa = {}
    cnt_aa = 0
    for p, q in product(range(spatial_dim), repeat=2):
        if p < q:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1

    for p, q, r, s in product(range(spatial_dim), repeat=4):
        if p < q and s < r:
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}
            # a_{q, \alpha}^{\dagger} -
            # a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(
                ((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))) - \
                                FermionOperator(((2 * p, 1), (2 * q, 1),
                                                 (2 * s, 0), (2 * r, 0))) - \
                                FermionOperator(((2 * q, 1), (2 * p, 1),
                                                 (2 * r, 0), (2 * s, 0))) + \
                                FermionOperator(((2 * q, 1), (2 * p, 1),
                                                 (2 * s, 0), (2 * r, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))

            for term in tpdm_element_as_pauli:
                pauli_tensor_list = sorted(list(
                    term.operations_as_set()), key=lambda x: x[0])
                rev_order_pauli_tensor_list = list(
                    map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                    d2_aa[bas_aa[(p, q)], bas_aa[(s, r)]] += pauli_to_coeff[
                                                        pauli_term.id()] * \
                                                        pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix.")
    return d2_aa


def pauli_terms_for_tpdm_bb(spatial_dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the beta-beta
    block of the 2-RDM

    Given a dictionary of expected values of Pauli terms, populate the
    beta-beta block of the 2-RDM

    :param Int spatial_dim: Dimension of the spatial-orbital basis.
    :param transform: fermion-to-qubit transform from OpenFermion
    :return: List of PauliTerms that corresponds to set of pauli terms to
             measure to construct the 2-RDM.
    """
    pauli_terms_to_measure = []
    pauli_to_rdm = {}
    for p, q, r, s in product(range(spatial_dim), repeat=4):
        if p < q and s < r:
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}
            # a_{q, \alpha}^{\dagger} -
            # a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1),
                                                 (2 * r + 1, 0),
                                                 (2 * s + 1, 0))) - \
                                FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1),
                                                 (2 * s + 1, 0),
                                                 (2 * r + 1, 0))) - \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1),
                                                 (2 * r + 1, 0),
                                                 (2 * s + 1, 0))) + \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1),
                                                 (2 * s + 1, 0),
                                                 (2 * r + 1, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))
            for term in tpdm_element_as_pauli:
                pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()),
                                   key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(
            map(lambda x: (x[1], x[0]), pauli_tensor_list))
        pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                         coefficient=term.coefficient)

        if pauli_term.id() not in pauli_to_rdm.keys():
            pauli_to_rdm[pauli_term.id()] = pauli_term
        else:
            if (abs(pauli_to_rdm[pauli_term.id()].coefficient) <
                    abs(pauli_term.coefficient)):
                pauli_to_rdm[pauli_term.id()] = pauli_term

    return list(pauli_to_rdm.values())


def pauli_to_tpdm_bb(spatial_dim, pauli_to_coeff, transform=jordan_wigner):
    """
    Populate the beta-beta block of the 2-RDM

    :param Int spatial_dim: spatial basis set rank
    :param pauli_to_coeff: a map between the Pauli term label to the expected
                           value.
    :param transform: Openfermion fermion-to-qubit transform
    :return: the 2-RDM alpha-beta block of the 2-RDM
    """
    aa_dim = int(spatial_dim * (spatial_dim - 1) / 2)
    d2_aa = np.zeros((aa_dim, aa_dim), dtype=complex)

    # build basis lookup table
    bas_aa = {}
    cnt_aa = 0
    for p, q in product(range(spatial_dim), repeat=2):
        if p < q:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1

    for p, q, r, s in product(range(spatial_dim), repeat=4):
        if p < q and s < r:
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}
            # a_{q, \alpha}^{\dagger} -
            # a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1),
                                                 (2 * r + 1, 0),
                                                 (2 * s + 1, 0))) - \
                                FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1),
                                                 (2 * s + 1, 0),
                                                 (2 * r + 1, 0))) - \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1),
                                                 (2 * r + 1, 0),
                                                 (2 * s + 1, 0))) + \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1),
                                                 (2 * s + 1, 0),
                                                 (2 * r + 1, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))

            for term in tpdm_element_as_pauli:
                pauli_tensor_list = sorted(list(
                    term.operations_as_set()), key=lambda x: x[0])
                rev_order_pauli_tensor_list = list(
                    map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                    d2_aa[bas_aa[(p, q)], bas_aa[(s, r)]] += pauli_to_coeff[
                                                        pauli_term.id()] * \
                                                        pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix.")
    return d2_aa


def pauli_terms_for_tpdm(dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the 2-RDM

    :param Int dim: Dimension of the spin-orbital basis.
    :param transform: fermion-to-qubit transform from OpenFermion
    :return: List of PauliTerms that corresponds to set of pauli terms to
             measure to construct the 2-RDM.
    """

    # first make a map between pauli terms and elements of the 2-RDM
    pauli_to_rdm = {}
    pauli_terms_to_measure = []
    for p, q, r, s in product(range(dim), repeat=4):
        if p != q and r != s:
            tpdm_element = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(tpdm_element)))
            for term in tpdm_element_as_pauli:
                pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()),
                                   key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(
            map(lambda x: (x[1], x[0]), pauli_tensor_list))
        pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                         coefficient=term.coefficient)

        if pauli_term.id() not in pauli_to_rdm.keys():
            pauli_to_rdm[pauli_term.id()] = pauli_term
        else:
            if (abs(pauli_to_rdm[pauli_term.id()].coefficient) <
                    abs(pauli_term.coefficient)):
                pauli_to_rdm[pauli_term.id()] = pauli_term

    return list(pauli_to_rdm.values())


def pauli_to_tpdm(dim, pauli_to_coeff, transform=jordan_wigner):
    """
    Construct the 2-RDM from expected values of Pauli operators

    Construct the 2-RDM by looping over the 2-RDM and loading up the
    coefficients for the expected values of each transformed Pauli operator.
    We assume the fermionic ladder operators are transformed via Jordan-Wigner.
    This constraint can be relaxed later.

    We don't check that the `pauli_expected` dictionary contains an
    informationally complete set of expected values.  This is useful for
    testing if under sampling the 2-RDM is okay if a projection technique is
    included in the calculation.

    :param Int dim: spin-orbital basis dimension
    :param Dict pauli_to_coeff: a map from pauli term ID's to
    :param func transform: optional argument defining how to transform
                           fermionic operators into Pauli operators
    """
    tpdm = np.zeros((dim, dim, dim, dim), dtype=complex)
    for p, q, r, s in product(range(dim), repeat=4):
        if p != q and r != s:
            tpdm_element = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(tpdm_element)))

            for term in tpdm_element_as_pauli:
                pauli_tensor_list = sorted(list(
                    term.operations_as_set()), key=lambda x: x[0])
                rev_order_pauli_tensor_list = list(
                    map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                    tpdm[p, q, r, s] += pauli_to_coeff[pauli_term.id()] * \
                                        pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix.")
    return tpdm
