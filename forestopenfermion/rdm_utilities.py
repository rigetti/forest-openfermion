"""
Utilities for RDMs
"""
import numpy as np
from itertools import product
from pyquil.paulis import PauliTerm, PauliSum


def get_sz_spin_adapted(measured_tpdm):
    """
    Take a spin-orbital 4-tensor 2-RDM and map to the SZ spin adapted version

    :param measured_tpdm: spin-orbital 2-RDM 4-tensor.  This accepts 2-RDMs
                          in OpenFermion format.
    :return: 2-RDM matrices for aa, bb, and ab
    """
    measured_tpdm = np.einsum('pqrs->pqsr', measured_tpdm)
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

    for i, j, k, l in product(range(dim // 2),
                              repeat=4):  # iterate over spatial indices
        d2_ab[bas_ab[(i, j)], bas_ab[(k, l)]] = measured_tpdm[
            2 * i, 2 * j + 1, 2 * k, 2 * l + 1].real

        if i < j and k < l:
            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[
                                            2 * i, 2 * j, 2 * k, 2 * l].real - \
                                                    measured_tpdm[
                                            2 * i, 2 * j, 2 * l, 2 * k].real - \
                                                    measured_tpdm[
                                            2 * j, 2 * i, 2 * k, 2 * l].real + \
                                                    measured_tpdm[
                                            2 * j, 2 * i, 2 * l, 2 * k].real

            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] = measured_tpdm[
                            2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1].real - \
                                                    measured_tpdm[
                            2 * i + 1, 2 * j + 1, 2 * l + 1, 2 * k + 1].real - \
                                                    measured_tpdm[
                            2 * j + 1, 2 * i + 1, 2 * k + 1, 2 * l + 1].real + \
                                                    measured_tpdm[
                            2 * j + 1, 2 * i + 1, 2 * l + 1, 2 * k + 1].real

            d2_aa[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5
            d2_bb[bas_aa[(i, j)], bas_aa[(k, l)]] *= 0.5

    return d2_aa, d2_bb, d2_ab


def unspin_adapt(d2aa, d2bb, d2ab):
    """
    Transform a sz_spin-adapted set of 2-RDMs back to the spin-orbtal 2-RDM

    :param d2aa: alpha-alpha block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2bb: beta-beta block of the 2-RDM.  Antisymmetric basis functions
                 are assumed for this block. block size is r_{s} * (r_{s} - 1)/2
                 where r_{s} is the number of spatial basis functions
    :param d2ab: alpha-beta block of the 2-RDM. no symmetry adapting is
                 perfomred on this block.  Map directly back to spin-orbital
                 components. This block should have linear dimension r_{s}^{2}
                 where r_{S} is the number of spatial basis functions.
    :return: four-tensor representing the spin-orbital density matrix in
             OpenFermion ordering.
    """
    sp_dim = int(np.sqrt(d2ab.shape[0]))
    so_dim = 2 * sp_dim
    tpdm = np.zeros((so_dim, so_dim, so_dim, so_dim), dtype=complex)

    # build basis look up table
    bas_aa = {}
    bas_ab = {}
    cnt_aa = 0
    cnt_ab = 0
    for p, q in product(range(sp_dim), repeat=2):
        if q > p:
            bas_aa[(p, q)] = cnt_aa
            cnt_aa += 1
        bas_ab[(p, q)] = cnt_ab
        cnt_ab += 1

    # map the d2aa and d2bb back to the spin-orbital 2-RDM
    for p, q, r, s in product(range(sp_dim), repeat=4):
        if p < q and r < s:
            tpdm[2 * p, 2 * q, 2 * r, 2 * s] = 0.5 * d2aa[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p, 2 * q, 2 * s, 2 * r] = -0.5 * d2aa[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * r, 2 * s] = -0.5 * d2aa[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q, 2 * p, 2 * s, 2 * r] = 0.5 * d2aa[
                bas_aa[(p, q)], bas_aa[(r, s)]]

            tpdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = 0.5 * d2bb[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * p + 1, 2 * q + 1, 2 * s + 1, 2 * r + 1] = -0.5 * d2bb[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * r + 1, 2 * s + 1] = -0.5 * d2bb[
                bas_aa[(p, q)], bas_aa[(r, s)]]
            tpdm[2 * q + 1, 2 * p + 1, 2 * s + 1, 2 * r + 1] = 0.5 * d2bb[
                bas_aa[(p, q)], bas_aa[(r, s)]]

        tpdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = d2ab[
            bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * r, 2 * s + 1] = -1 * d2ab[
            bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * p, 2 * q + 1, 2 * s + 1, 2 * r] = -1 * d2ab[
            bas_ab[(p, q)], bas_ab[(r, s)]]
        tpdm[2 * q + 1, 2 * p, 2 * s + 1, 2 * r] = d2ab[
            bas_ab[(p, q)], bas_ab[(r, s)]]

    return np.einsum('pqrs->pqsr', tpdm)


########################################
#
# Utilities for label mapping to qubits
#
########################################
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
        relabeled_terms.append(PauliTerm.from_list(
            new_term_as_list, coefficient=term.coefficient))
    return relabeled_terms


def pauli_term_from_string(pauli_string):
    """
    Convert a string to a Pauli term

    Strings look like `X0Z1Y3I4'

    :param pauli_string: String to be translated to a PauliTerm
    :return: PauliTerm
    """
    pauli_elements = pauli_string[::2]
    qubit_elements = list(map(int, pauli_string[1::2]))
    return PauliTerm.from_list(list(zip(pauli_elements, qubit_elements)))


def pauli_dict_relabel(pauli_dict, label_map):
    """
    Relabel the elements of a dictionary where labels are Pauli ID's

    :param pauli_dict: pauli dict to relabel.
    :param label_map: a dictionary mapping old label to new label
    :return: a list of pauli_terms relabeled
    """
    relabeled_dict = {}
    for key, value in pauli_dict.items():
        if key == '':  # identity term
            relabeled_dict[''] = value
        else:
            term = pauli_term_from_string(key)
            relabed_term = pauli_term_relabel(term, label_map=label_map)[0]
            relabeled_dict[relabed_term.id()] = value

    return relabeled_dict
