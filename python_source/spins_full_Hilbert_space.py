import numpy as np

from states import (
    ρ_from_ket,
)  # TODO: add path to pleasantPheasant? Or rebuild from scratch with the symmeterised?
from operators import dag, exp_matrix

σx = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]])
σy = np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]])
σz = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]])
σplus = σx + 1j * σy
σminus = σx - 1j * σy


def binary_to_index(M, binary):
    """Convert binary string to index, e.g. '1010001110' to 369, via reverse lexicographic order."""
    assert len(binary) == M
    return 2**M - 1 - int(binary, 2)


def index_to_binary(M, index):
    """Inverse of binary_to_index."""
    return bin(2**M - 1 - index).split("b")[1].zfill(M)


def spin_state_type(M, state):
    if state.shape == (2**M,) and state.dtype == np.dtype("complex128"):
        return "ket"
    elif state.shape == (2**M, 2**M) and state.dtype == np.dtype("complex128"):
        return "DM"
    else:
        raise ValueError("Unknown state type.")


def Jz_probabilities(M, state):
    """Returns Jz probabilities of state."""
    stype = spin_state_type(M, state)
    if stype == "ket":
        return np.abs(state) ** 2
    elif stype == "DM":
        return state.diagonal().real
    else:
        raise ValueError("Unknown state type.")


def spin_state_norm(M, state):
    """Returns normalisation of spin state."""
    stype = spin_state_type(M, state)
    if stype == "ket":
        return (state.conj() @ state).real
    elif stype == "DM":
        return np.trace(state).real
    else:
        raise ValueError("Unknown state type.")


def normalise_spin_state(M, state):
    """Return normalised spin state."""
    return state / spin_state_norm(M, state) ** 0.5


def σz_eigenstate(M, binary, return_ket=False):
    """Ket or DM of M spin-1/2 systems in the σz basis in reverse lexicographic order."""
    ket = np.zeros(2**M, dtype=complex)
    ket[binary_to_index(M, binary)] = 1
    if return_ket:
        return ket
    else:
        return ρ_from_ket(ket)


def Jz_up_state(M, return_ket=False):
    """|1,1,1,...,1>"""
    return σz_eigenstate(M, M * "1", return_ket=return_ket)


def Jz_down_state(M, return_ket=False):
    """|0,0,0,...,0>"""
    return σz_eigenstate(M, M * "0", return_ket=return_ket)


def GHZ(M, return_ket=False):
    """(|1,1,1,...,1> + |0,0,0,...,0>)/sqrt2"""
    ket = (
        Jz_up_state(M, return_ket=True) + Jz_down_state(M, return_ket=True)
    ) / np.sqrt(2)
    if return_ket:
        return ket
    else:
        return ρ_from_ket(ket)


def Jx_up_state(M, return_ket=False):
    """|↑_x>^{⊗M}"""
    # sum of every single permutation, TODO: define Jx_down_state by using the sign of the permutation
    ket = σz_eigenstate(M, index_to_binary(M, 0), return_ket=True)
    for j in range(1, 2**M):
        ket += σz_eigenstate(M, index_to_binary(M, j), return_ket=True)
    ket *= 2 ** (-M / 2)
    if return_ket:
        return ket
    else:
        return ρ_from_ket(ket)
        # return np.fill((2**M, 2**M), 2**(-M))


def zero_ket(M):
    """Empty ket to construct states."""
    return np.zeros(2**M, dtype=complex)


def symmeterised_Jz_eigenstate(M, Jz_eigenvalue, return_ket=False):
    # TODO: determine if these are the Dicke states, see below Ma+11's Eq.18. Is J^2 fixed for a given M by particle conservation? Simon said that J^2 is like n^2?
    # Jz_eigenvalue = -M/2, -M/2+1, ..., M/2-1, M/2
    ket = zero_ket(M)
    num_terms = 0
    for j in range(2**M):
        binary = index_to_binary(M, j)
        if 0.5 * M - binary.count("0") == Jz_eigenvalue:
            ket += σz_eigenstate(M, binary, return_ket=True)
            num_terms += 1
    if num_terms == 0:
        raise ValueError("Jz_eigenvalue must be a valid eigenvalue.")
    ket *= num_terms ** (-0.5)
    if return_ket:
        return ket
    else:
        return ρ_from_ket(ket)


def symmeterised_Jz_probabilities(M, state):
    """Returns Jz symmeterised probabilities of state, see below Ma+11's Eq.19."""
    stype = spin_state_type(M, state)
    if stype == "ket":
        data = np.zeros((2, M + 1))
        for j in range(M + 1):
            eigenvalue = -M / 2 + j
            dicke_ket = symmeterised_Jz_eigenstate(M, eigenvalue, return_ket=True)
            probability = (dicke_ket.conj() @ state).real ** 2
            data[:, j] = eigenvalue, probability
        return data
    elif stype == "DM":
        data = np.zeros((2, M + 1))
        for j in range(M + 1):
            eigenvalue = -M / 2 + j
            dicke_DM = symmeterised_Jz_eigenstate(M, eigenvalue, return_ket=False)
            probability = np.trace(dicke_DM @ state).real ** 2
            data[:, j] = eigenvalue, probability
        return data
    else:
        raise ValueError("Unknown state type.")


def zero_matrix(M):
    """Zero matrix to construct operators."""
    return np.full((2**M, 2**M), 0, dtype=complex)


def Jz(M):
    op = zero_matrix(M)
    for j in range(2**M):
        op[j, j] = 0.5 * M - index_to_binary(M, j).count("0")
    return op


def bit_flip(binary, index_to_flip):
    """Return binary with bit at index_to_flip flipped."""
    lb = list(binary)
    lb[index_to_flip] = str(1 - int(binary[index_to_flip]))
    return "".join(lb)


def Jx(M):
    op = zero_matrix(M)
    for j in range(2**M):
        for k in range(2**M):
            a = index_to_binary(M, j)
            for l in range(M):
                b0 = index_to_binary(M, k)
                b = bit_flip(b0, l)
                if a == b:
                    op[j, k] += 0.5
    return op


def sign_from_bit(bit):
    if bit == "1":
        return 1
    elif bit == "0":
        return -1
    else:
        raise ValueError("Bit must be 0 or 1.")


def Jy(M):
    op = zero_matrix(M)
    for j in range(2**M):
        for k in range(2**M):
            a = index_to_binary(M, j)
            for l in range(M):
                b0 = index_to_binary(M, k)
                b = bit_flip(b0, l)
                if a == b:
                    op[j, k] += 0.5 * sign_from_bit(b0[l]) * 1j
    return op


def Jplus(M):
    """Jump operator of the collective amplitude damping channel.

    Note that σplus and σminus have their only element equal to 2.
    """
    return Jx(M) + 1j * Jy(M)


def Jminus(M):
    """Jump operator of the collective amplitude damping channel."""
    return Jx(M) - 1j * Jy(M)


def commutator(A, B):
    return A @ B - B @ A


def Jy_via_commutator(M):
    """[σz, σx] = 2 i σy, [Jz, Jx] = i Jy"""
    return -1j * commutator(Jz(M), Jx(M))


def rotate(M, state, θ, ϕ, truncation=None):
    """Return rotated spin state, see Eq.23 of Ma+11."""
    ζ = -0.5 * θ * np.exp(-1j * ϕ)
    A = ζ * Jplus(M) - ζ.conj() * Jminus(M)
    U = exp_matrix(A, truncation=truncation)
    stype = spin_state_type(M, state)
    if stype == "ket":
        return U @ state
    elif stype == "DM":
        return U @ state @ dag(U)


def CSS(M, θ, ϕ, return_ket=False, truncation=None):
    """Coherent spin state, |θ, ϕ>, see Eq.23 of Ma+11."""
    state = Jz_up_state(M, return_ket=return_ket)
    return rotate(M, state, θ, ϕ, truncation=truncation)


def TAT(M, state, χ, truncation=None):
    """Returns the two-axis twisted state."""
    # TODO: check phase of \chi
    H = (
        -0.5
        * 1j
        * χ
        * (np.linalg.matrix_power(Jplus(M), 2) - np.linalg.matrix_power(Jminus(M), 2))
    )
    U = exp_matrix(-1j * H, truncation=truncation)

    # TODO: normalise state after unitary since appears not to be normalised? Should be unitary if truncation=None?
    stype = spin_state_type(M, state)
    if stype == "ket":
        return U @ state
    elif stype == "DM":
        return U @ state @ dag(U)


def OAT_along_z(M, state, χ, truncation=None):
    """Returns the one-axis twisted state along z."""
    # TODO: check phase of \chi
    H = χ * np.linalg.matrix_power(Jz(M), 2)
    U = exp_matrix(H, truncation=truncation)

    stype = spin_state_type(M, state)
    if stype == "ket":
        return U @ state
    elif stype == "DM":
        return U @ state @ dag(U)


def OAT_along_x(M, state, χ, truncation=None):
    """Returns the one-axis twisted state along x."""
    # TODO: check phase of \chi
    H = χ * np.linalg.matrix_power(Jx(M), 2)
    U = exp_matrix(H, truncation=truncation)

    stype = spin_state_type(M, state)
    if stype == "ket":
        return U @ state
    elif stype == "DM":
        return U @ state @ dag(U)


def supremum_matrix_distance(A, B):
    return np.abs(A - B).max()


def are_equal(A, B, ε=1e-15):
    return supremum_matrix_distance(A, B) <= ε


def is_hermitian(A, ε=1e-15):
    return are_equal(A, dag(A), ε)


def is_unitary(M, A, ε=1e-15):
    I = np.identity(2**M)
    return are_equal(A @ dag(A), I, ε) & are_equal(dag(A) @ A, I, ε)
