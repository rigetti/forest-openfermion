"""
A module for measuring 1-RDMs and 2-RDMs from a quantum resource by
Monte Carlo averaging
"""
import numpy as np
from itertools import product
from grove.measurements.estimation import remove_imaginary_terms
from pyquil.paulis import PauliTerm, PauliSum

from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from forestopenfermion.pyquil_connector import qubitop_to_pyquilpauli


def pauli_term_relabel(pauli_sum, label_map):
    """
    Relabel the elements of a pauli_sum via the `label_map`

    :param pauli_sum: pauli sum to relabel.  this can be a PauliTerm, PauliSum,
                      or even better a LIST!
    :param label_map: a dictionary mapping old label to new label
    :return: a list of pauli_terms relabeled
    """
    if isinstance(pauli_sum, PauliTerm):
        pauli_sum = PauliSum([pauli_sum])

    if isinstance(pauli_sum, PauliSum):
        pauli_sum = pauli_sum.terms

    relabeled_terms = []
    for term in pauli_sum:
        new_term_as_list = []
        for qlabel, pauli_element in term._ops.items():
            new_term_as_list.append((pauli_element, label_map[qlabel]))
        relabeled_terms.append(PauliTerm.from_list(new_term_as_list,
                                                   coefficient=term.coefficient))
    return relabeled_terms


def pauli_terms_for_tpdm_ab(spatial_dim, transform=jordan_wigner):
    """
    Build the alpha-beta block of the 2-RDM

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
        spin_adapted_term = FermionOperator(((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)))

        tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))
        for term in tpdm_element_as_pauli:
            pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()), key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
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

    :param dim: spatial basis set rank
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
        spin_adapted_term = FermionOperator(((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s, 0)))

        tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))

        for term in tpdm_element_as_pauli:
           pauli_tensor_list = sorted(list(
               term.operations_as_set()), key=lambda x: x[0])
           rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
           pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                            coefficient=term.coefficient)
           try:
              d2_ab[bas_ab[(p, q)], bas_ab[(s, r)]] += pauli_to_coeff[pauli_term.id()] * \
                                                       pauli_term.coefficient
           except KeyError:
               raise Warning("key was not in the coeff matrix. 2-RDM is " +
                             "not informationally complete")
    return d2_ab


def pauli_terms_for_tpdm_aa(spatial_dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the alpha-alpha
    block of the 2-RDM

    :param Int sdim: Dimension of the spatial-orbital basis.
    :return: :ist of PauliTerms that corresponds to set of pauli terms to
             measure to construct the 2-RDM.
    :param spatial_dim:
    :param transform:
    :return:
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
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}a_{q, \alpha}^{\dagger} - a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))) - \
                                FermionOperator(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0))) - \
                                FermionOperator(((2 * q, 1), (2 * p, 1), (2 * r, 0), (2 * s, 0))) + \
                                FermionOperator(((2 * q, 1), (2 * p, 1), (2 * s, 0), (2 * r, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))
            for term in tpdm_element_as_pauli:
                pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()), key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
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
    :param dim:
    :param pauli_to_coeff:
    :param transform:
    :return:
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
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}a_{q, \alpha}^{\dagger} - a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0))) - \
                                FermionOperator(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0))) - \
                                FermionOperator(((2 * q, 1), (2 * p, 1), (2 * r, 0), (2 * s, 0))) + \
                                FermionOperator(((2 * q, 1), (2 * p, 1), (2 * s, 0), (2 * r, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))

            for term in tpdm_element_as_pauli:
                pauli_tensor_list = sorted(list(
                    term.operations_as_set()), key=lambda x: x[0])
                rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                   d2_aa[bas_aa[(p, q)], bas_aa[(s, r)]] += pauli_to_coeff[pauli_term.id()] * \
                                                            pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix. 2-RDM is " +
                                  "not informationally complete")
    return d2_aa


def pauli_terms_for_tpdm_bb(spatial_dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the beta-beta
    block of the 2-RDM

    :param Int sdim: Dimension of the spatial-orbital basis.
    :return: :ist of PauliTerms that corresponds to set of pauli terms to
             measure to construct the 2-RDM.
    :param spatial_dim:
    :param transform:
    :return:
    """
    pauli_terms_to_measure = []
    pauli_to_rdm = {}
    for p, q, r, s in product(range(spatial_dim), repeat=4):
        if p < q and s < r:
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}a_{q, \alpha}^{\dagger} - a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))) - \
                                FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0))) - \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))) + \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))
            for term in tpdm_element_as_pauli:
                pauli_terms_to_measure.append(term)

    for term in pauli_terms_to_measure:
        # convert term into numerically order pauli tensor term
        pauli_tensor_list = sorted(list(term.operations_as_set()), key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
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
    :param spatial_dim:
    :param pauli_to_coeff:
    :param transform:
    :return:
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
            # generator 1/sqrt(2) * (a_{p, \alpha}^{\dagger}a_{q, \alpha}^{\dagger} - a_{p, \beta}^{\dagger}a_{q, \beta}^{\dagger})
            spin_adapted_term = FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))) - \
                                FermionOperator(((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0))) - \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1), (2 * r + 1, 0), (2 * s + 1, 0))) + \
                                FermionOperator(((2 * q + 1, 1), (2 * p + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)))
            spin_adapted_term *= 0.5

            tpdm_element_as_pauli = remove_imaginary_terms(
                qubitop_to_pyquilpauli(transform(spin_adapted_term)))

            for term in tpdm_element_as_pauli:
                pauli_tensor_list = sorted(list(
                    term.operations_as_set()), key=lambda x: x[0])
                rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                   d2_aa[bas_aa[(p, q)], bas_aa[(s, r)]] += pauli_to_coeff[pauli_term.id()] * \
                                                            pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix. 2-RDM is " +
                                  "not informationally complete")
    return d2_aa


def pauli_terms_for_tpdm(dim, transform=jordan_wigner):
    """
    Generate a set of pauli operators to measure to evaluate the 2-RDM

    :param Int dim: Dimension of the spin-orbital basis used to construct the
                    2-RDM.
    :return: :ist of PauliTerms that corresponds to set of pauli terms to
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
        pauli_tensor_list = sorted(list(term.operations_as_set()), key=lambda x: x[0])
        rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
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
    :return:
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
                rev_order_pauli_tensor_list = list(map(lambda x: (x[1], x[0]), pauli_tensor_list))
                pauli_term = PauliTerm.from_list(rev_order_pauli_tensor_list,
                                                 coefficient=term.coefficient)
                try:
                   tpdm[p, q, r, s] += pauli_to_coeff[pauli_term.id()] * \
                                       pauli_term.coefficient
                except KeyError:
                    raise Warning("key was not in the coeff matrix. 2-RDM is " +
                                  "not informationally complete")
    return tpdm
