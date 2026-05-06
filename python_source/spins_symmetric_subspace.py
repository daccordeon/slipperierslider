import numpy as np
from scipy.linalg import expm, logm, sqrtm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tkr

π = np.pi


def dag(A):
    """Hermitian conjugate"""
    return A.conj().T


# NB: Do not use pow() as it will act element-wise.
power = np.linalg.matrix_power


# General Paulis
def I(D):
    return np.identity(D)


def Jp(D):
    """D is the dimension of the quDit."""
    Jp = np.zeros((D, D), dtype=complex)
    for j in range(D - 1):
        Jp[j, j + 1] = np.sqrt((j + 1) * (D - 1 - j))
    return Jp


def Jm(D):
    Jm = np.zeros((D, D), dtype=complex)
    for j in range(1, D):
        Jm[j, j - 1] = np.sqrt(j * (D - 1 - j + 1))
    return Jm


def Jx(D):
    return (Jp(D) + Jm(D)) / 2


def Jy(D):
    return (Jp(D) - Jm(D)) / 2j


def commutator(A, B):
    return A @ B - B @ A


def anticommutator(A, B):
    return A @ B + B @ A


def Jz(D):
    return -1j * commutator(Jx(D), Jy(D))


# 2D Paulis
σx = 2 * Jx(2)
σy = 2 * Jy(2)
σz = 2 * Jz(2)
σp = 2 * Jp(2)
σm = 2 * Jm(2)


# linear algebra
def matrix_vectorise(A):
    """A is a square matrix"""
    return A.T.reshape(len(A) ** 2)


def matrix_unvectorise(v):
    """v is a vector whose length is a square number"""
    return v.reshape(int(len(v) ** 0.5), int(len(v) ** 0.5)).T


# states, see also CSS below
def construct_ket_from_parts(D, parts):
    assert len(parts) == 2 * D
    re = parts[0::2]
    im = parts[1::2]
    return normalise(D, re + 1j * im)


def ρ_from_ket(ket):
    """Returns density matrix from ket."""
    return np.outer(ket, np.conjugate(ket))


def state_type(D, state):
    if state.shape == (D,) and state.dtype == np.dtype("complex128"):
        return "ket"
    elif state.shape == (D, D) and state.dtype == np.dtype("complex128"):
        return "density matrix"
    else:
        raise ValueError(
            f"Unknown state type. Dimension of state is {state.shape} but was given dimension D = {D}. Datatype is {state.dtype} (needs to be complex128)."
        )


def ρ_from_state(D, state):
    """Returns density matrix from state."""
    stype = state_type(D, state)
    if stype == "ket":
        return ρ_from_ket(state)
    elif stype == "density matrix":
        return state
    else:
        raise ValueError("Unknown state type.")


def norm(D, state):
    """Returns normalisation of spin state. This is Euclidean norm squared."""
    stype = state_type(D, state)
    if stype == "ket":
        return (state.conj() @ state).real
    elif stype == "density matrix":
        return np.trace(state).real ** 2
    else:
        raise ValueError("Unknown state type.")


def normalise(D, state):
    """Return normalised spin state."""
    return state / norm(D, state) ** 0.5


def purity(ρ):
    return np.trace(power(ρ, 2)).real


def rank(ρ, tol=1e-15, **kwargs):
    return np.linalg.matrix_rank(ρ, tol=tol, **kwargs)


def entropy(ρ, D=None, normalised=False):
    """von Neumann entropy of a density matrix
    https://en.wikipedia.org/wiki/Von_Neumann_entropy"""
    S = -np.trace(ρ @ logm(ρ)).real
    if normalised:
        assert D is not None
        return S / np.log(D)
    else:
        return S
    ## equivalent to the Shannon entropy of the eigenvalues as follows:
    # λs = np.linalg.eigh(ρ).eigenvalues
    # return -sum(λs * np.log(λs)).real


def generate_seed_sequence(seed, length):
    if type(seed) == np.random.bit_generator.SeedSequence:
        raise ValueError(
            "SeedSequence tracks changes between runs with the same object which may not be what you want. Use a different type of seed to be careful."
        )
    else:
        return np.random.SeedSequence(seed).spawn(length)


def random_ket(D, seed=None):
    if seed is None:
        re = 2 * np.random.rand(D) - 1
        im = 2 * np.random.rand(D) - 1
    else:
        rng = np.random.default_rng(seed)
        re = 2 * rng.random(D) - 1
        im = 2 * rng.random(D) - 1

    return normalise(D, re + 1j * im)


def ith_eigenstate(A, i):
    """ith eigenvector with the ith eigenvalue in ascending order."""
    # assert is_hermitian(A, ε=1e-9)
    λs, vs = np.linalg.eigh(A)
    return vs[:, i]


def extreme_eigenstate(A, max_or_min, return_ket=True):
    # From the docs (edited): The normalized (unit "length") eigenvectors, such that the column vs[:, i] is the eigenvector corresponding to the eigenvalue λs[i]. The eigenvalues are returned in ascending order.
    # assert is_hermitian(A, ε=1e-9)
    # print(f'Should be Hermitian, error: {supremum_matrix_distance(A, dag(A)):.2g}.')
    λs, vs = np.linalg.eigh(A)
    assert max_or_min in ["max", "min"]
    # i = -1 if max_or_min == 'max' else 0
    i = np.argmax(λs) if max_or_min == "max" else np.argmin(λs)
    ket = vs[:, i]
    if return_ket:
        return ket
    else:
        return ρ_from_ket(ket)


def maximum_eigenstates(A, round_to=5, add_more=0):
    """Return a list of basis vectors for the eigenspace of A associated with the largest eigenvalue. I assume that A \geq 0."""
    # normalise
    A /= np.max(A)
    # round
    A = A.round(round_to)
    λs, vs = np.linalg.eigh(A)
    if add_more == 0:
        return vs[:, λs == np.max(λs)].T
    else:
        # return all of the largest eigenspace plus add_more next largest eigenvalues
        num_in_largest = len(λs[λs == np.max(λs)])
        # λs are sorted in ascending order
        return vs[:, -num_in_largest - add_more :].T


def cat_state(D, A, θ):
    ket1 = extreme_eigenstate(A, "max")
    ket2 = extreme_eigenstate(A, "min")
    ket = (ket1 + np.exp(1j * θ) * ket2) / np.sqrt(2)
    return normalise(D, ket)


def Jz_up_state(D, return_ket=True):
    return extreme_eigenstate(Jz(D), "max", return_ket)


def Jz_down_state(D, return_ket=True):
    """Also called the vacuum state."""
    return extreme_eigenstate(Jz(D), "min", return_ket)


vacuum = Jz_down_state


def fidelity(D, state1, state2):
    """Bures quantum fidelity between states ρ1 and ρ2, sometimes given as the square root of this quantity, but apparantely this convention is more common, see https://en.wikipedia.org/wiki/Fidelity_of_quantum_states."""
    ρ1 = ρ_from_state(D, state1)
    ρ2 = ρ_from_state(D, state2)
    return (np.trace(sqrtm(sqrtm(ρ2) @ ρ1 @ sqrtm(ρ2)))).real ** 2


# more operators methods
def Hermitian_basis(D):
    """Return a Hermitian basis for D-by-D operators."""
    basis = []
    for i in range(D):
        for j in range(i, D):
            if i == j:
                h = np.zeros((D, D), dtype=complex)
                h[i, i] = 1
                basis.append(h)
            else:
                h = np.zeros((D, D), dtype=complex)
                h[i, j] = h[j, i] = 1
                basis.append(h)

                h = np.zeros((D, D), dtype=complex)
                h[i, j] = 1j
                h[j, i] = -1j
                basis.append(h)
    basis = np.array(basis)
    return basis


def ip(A, B):
    """Returns the trace inner produce of A and B, Tr(A†B)."""
    return np.trace(dag(A) @ B)


def op_from_pv(pv):
    # new_op = np.sum([k * σ for k, σ in zip(pv, [I(2), σx, σy, σz])], axis=0)
    A = np.zeros((2, 2), dtype=complex)
    for j in range(4):
        A += pv[j] * np.array([I(2), σx, σy, σz])[j]
    return A


def Pauli_vector(A, non_Hermitian=False):
    """assert are_equal(np.dot(Pauli_vector(reL1), Pauli_vector(imL1)), ip(reL1, imL1) / 2)"""
    # the factor of 1/2 comes from compensating tr(I) = 2
    if not non_Hermitian:
        return np.array([ip(A, σ) for σ in [I(2), σx, σy, σz]]) / 2
    else:
        re_pv, im_pv = Pauli_vector(re_op(A)), Pauli_vector(im_op(A))
        assert are_equal(op_from_pv(re_pv + 1j * im_pv), A)
        return re_pv, im_pv


def bloch_vector(ket):
    """TODO: combine with Pauli_vector() via stype"""
    # https://en.wikipedia.org/wiki/Bloch_sphere#Definition
    ρ = ρ_from_ket(ket)
    # np.linalg.norm of this vector is related to the state's purity
    return 2 * Pauli_vector(ρ)[1:].real


# Gram-Schmidt process, general methods
# TODO: detect the stype automatically and apply accordingly
def ip_general(A, B, stype):
    if stype == "operator":
        return np.trace(dag(A) @ B)
    elif stype == "ket":
        return np.dot(A.conj(), B)
    elif stype == "real vector":
        return np.dot(A, B)
    else:
        raise ValueError(f"stype = {stype} not recognised.")


def orthogonal_component_general(A, B, stype):
    """Returns the part of A that is orthogonal to B."""
    norm_sqr_B = ip_general(B, B, stype)
    if norm_sqr_B == 0:
        return A
    else:
        return A - ip_general(A, B, stype).conj() / norm_sqr_B * B


def orthogonal_component_from_orthogonal_set_general(A, S, stype):
    """Returns the part of A that is orthogonal to all B in orthogonal list S."""
    Aperp = A
    for B in S:
        Aperp = orthogonal_component_general(Aperp, B, stype)
    return Aperp


def orthogonalise_set_general(S, stype, normalisation_fn=None, D=None):
    S_orth = []
    for A in S:
        Aperp = orthogonal_component_from_orthogonal_set_general(A, S_orth, stype)
        tol = 1e-15
        if np.all(np.abs(Aperp) < tol):
            continue
        else:
            S_orth.append(Aperp)
    if normalisation_fn is not None:
        if normalisation_fn is True:
            normalisation_fn = lambda A: normalise(D, A)
        S_orth = [normalisation_fn(A) for A in S_orth]
    return np.array(S_orth) if type(S) is np.ndarray else S_orth


def orthogonal_component_from_set_general(A, S, stype):
    """Returns the part of A that is orthogonal to all B in list S."""
    S_orth = orthogonalise_set_general(S, stype)
    return orthogonal_component_from_orthogonal_set_general(A, S_orth, stype)


def is_this_set_orthogonal_general(S, stype, tol=1e-10):
    for i, A in enumerate(S):
        for B in S[:i]:
            if np.abs(ip_general(A, B, stype)) > tol:
                return False
    else:
        return True


# # TODO: refactor these common Gram-Schmidt methods
# def orthogonal_component(A, B):
#     """Returns the part of A that is orthogonal to B such that Tr(A†B)=0."""
#     if ip(B, B) == 0:
#         return A
#     else:
#         return A - ip(A, B).conj() / ip(B, B) * B

# def orthogonal_component_from_set(A, S):
#     """Returns the part of A that is orthogonal to all B in list S."""
#     Aperp = A
#     for i, B in enumerate(S):
#         Bperp = orthogonal_component_from_set(B, S[:i])
#         Aperp = orthogonal_component(Aperp, Bperp)
#     return Aperp

# def orthogonal_component_from_orthogonal_set(A, S):
#     """Returns the part of A that is orthogonal to all B in orthogonal list S."""
#     Aperp = A
#     for B in S:
#         Aperp = orthogonal_component(Aperp, B)
#     return Aperp

# def orthogonalise_set_of_operators(S):
#     S_orth = []
#     for A in S:
#         Aperp = orthogonal_component_from_orthogonal_set(A, S_orth)
#         tol = 1e-15
#         if np.all(np.abs(Aperp) < tol):
#             continue
#         else:
#             S_orth.append(Aperp)
#     return np.array(S_orth) if type(S) is np.ndarray else S_orth

# # TODO: refactor these Gram-Schmidt methods, 3 different families of them
# def orthogonal_component_kets(ket1, ket2):
#     """Returns the part of A that is orthogonal to B."""
#     return ket1 - np.dot(ket1.conj(), ket2).conj() / np.dot(ket2.conj(), ket2) * ket2

# def orthogonal_component_kets_from_set(ket1, S):
#     ket1_perp = ket1
#     for i, ket2 in enumerate(S):
#         ket2_perp = orthogonal_component_kets_from_set(ket2, S[:i])
#         ket1_perp = orthogonal_component_kets(ket1_perp, ket2_perp)
#     return ket1_perp

# def orthogonal_component_kets_from_orthogonal_set(ket1, S):
#     ket1_perp = ket1
#     for ket2 in S:
#         ket1_perp = orthogonal_component_kets(ket1_perp, ket2)
#     return ket1_perp

# def orthogonalise_set_of_kets(D, S, please_normalise=True):
#     S_orth = []
#     for ket1 in S:
#         ket1_perp = orthogonal_component_kets_from_orthogonal_set(ket1, S_orth)
#         # drop ket if zero, i.e. in the span of the others, or zero within some numerical tolerance
#         tol = 1e-15
#         if np.all(np.abs(ket1_perp) < tol):
#             continue
#         else:
#             S_orth.append(ket1_perp)
#     if please_normalise:
#         S_orth = [normalise(D, v) for v in S_orth]
#     return np.array(S_orth) if type(S) is np.ndarray else S_orth

# def is_this_set_orthogonal(S, tol=1e-10):
#     for i, ket1 in enumerate(S):
#         for ket2 in S[:i]:
#             if np.abs(np.dot(ket1.conj(), ket2)) > tol:
#                 return False
#     else:
#         return True


def furthest_apart_complex_numbers(zs, return_indices=False):
    dist = z1_opt = z2_opt = i_opt = j_opt = 0
    for i, z1 in enumerate(zs):
        for j, z2 in enumerate(zs[:i]):
            dist0 = np.abs(z1 - z2)
            if dist0 > dist:
                dist = dist0
                z1_opt, z2_opt = z1, z2
                i_opt, j_opt = i, j

    if return_indices:
        return i_opt, j_opt
    else:
        return z1_opt, z2_opt


# random operators
def random_op(D, seed=None):
    """Returns an operator with elements z sampled uniformly from the square in the complex plane with -1 < Re[z] < 1 and -1 < Im[z] < 1."""
    if seed is None:
        re = 2 * np.random.rand(D, D) - 1
        im = 2 * np.random.rand(D, D) - 1
    else:
        rng = np.random.default_rng(seed)
        re = 2 * rng.random((D, D)) - 1
        im = 2 * rng.random((D, D)) - 1

    return re + 1j * im


def re_op(A):
    return (A + dag(A)) / 2


def im_op(A):
    return (A - dag(A)) / 2j


def parts_op(A):
    return re_op(A), im_op(A)


def random_Hermitian_op(D, seed=None):
    A = random_op(D, seed)
    return re_op(A)


def random_unitary(D, seed=None):
    H = random_Hermitian_op(D, seed)
    return expm(-1j * H)


# Paulis
def orthogonal_Pauli_operator(D, σ1, σ2):
    """σ1 and σ2 are assumed traceless."""
    pv1 = Pauli_vector(σ1)[1:]
    pv2 = Pauli_vector(σ2)[1:]
    new_Pauli_vector = np.cross(pv1, pv2)
    # TODO: Find a nice numpy tensor contraction way to do this.
    new_op = np.sum([k * σj for k, σj in zip(new_Pauli_vector, [σx, σy, σz])], axis=0)
    new_op /= np.sqrt(ip(new_op, new_op))
    return new_op


def σpm_representation(D, L1, L2=None):
    """Return a, b such that M^{-1} @ L1 @ M = a * σp + b * σm, where M is a change of basis matrix."""
    # re/im parts of L1
    reL1 = (L1 + dag(L1)) / 2
    imL1 = (L1 - dag(L1)) / 2j

    # L1prime is the Hermitian traceless operator which is orthogonal to the re/im parts of L1
    L1prime = orthogonal_Pauli_operator(D, reL1, imL1)

    # columns of vs are the eigenvectors, vs is therefore the matrix that transform from e1,e2 to v1,v2
    vs = np.linalg.eigh(L1prime).eigenvectors
    change_of_basis = lambda M: np.linalg.inv(vs) @ M @ vs
    L1_transformed = change_of_basis(L1)
    a = ip(L1_transformed, σp).conj() / 4
    b = ip(L1_transformed, σm).conj() / 4
    if L2 is None:
        return dict(a=a, b=b)
    else:
        L2_transformed = change_of_basis(L2)
        return dict(
            a=a,
            b=b,
            L2_transformed=L2_transformed,
            vs=vs,
            change_of_basis=change_of_basis,
        )


# particular exponentials
def exp_matrix(exponent, truncation=None):
    """Returns exp(exponent).

    Args:
        exponent (matrix)
        truncation (None or int): truncation index
    """
    if truncation is None:
        return expm(exponent)
    else:
        summands = (
            1 / np.math.factorial(i) * power(exponent, i) for i in range(truncation + 1)
        )
        return np.sum(summands)


def apply(D, U, state):
    """Apply unitary operator U to state."""
    stype = state_type(D, state)
    if stype == "ket":
        return U @ state
    elif stype == "density matrix":
        return U @ state @ dag(U)


def rotate(D, state, θ, ϕ, truncation=None):
    """Return rotated spin state, see Eq.23 of Ma+11."""
    ζ = -0.5 * θ * np.exp(-1j * ϕ)
    A = ζ * Jp(D) - ζ.conj() * Jm(D)
    U = exp_matrix(A, truncation=truncation)
    return apply(D, U, state)


def CSS(D, θ, ϕ, return_ket=True, truncation=None):
    """Coherent spin state, |θ, ϕ>, see Eq.23 of Ma+11."""
    state = Jz_up_state(D, return_ket=return_ket)
    return rotate(D, state, θ, ϕ, truncation=truncation)


def TAT(D, state, χ, truncation=None):
    """Returns the two-axis twisted state."""
    # TODO: check phase of \chi, when does it revive? Is it at pi?
    H = -0.5 * 1j * χ * (power(Jp(D), 2) - power(Jm(D), 2))
    U = exp_matrix(-1j * H, truncation=truncation)
    # TODO: normalise state after unitary since appears not to be normalised? Should be unitary if truncation=None?
    return apply(D, U, state)


def OAT_along_z(D, state, χ, truncation=None):
    """Returns the one-axis twisted state along z. Ma+11 Eqs.128,130."""
    # TODO: check phase of \chi
    H = χ * power(Jz(D), 2)
    U = exp_matrix(-1j * H, truncation=truncation)
    return apply(D, U, state)


def OAT_along_x(D, state, χ, truncation=None):
    """Returns the one-axis twisted state along x."""
    # TODO: check phase of \chi
    H = χ * power(Jx(D), 2)
    U = exp_matrix(-1j * H, truncation=truncation)
    return apply(D, U, state)


# EVs


def EV(D, state, A):
    stype = state_type(D, state)
    if stype == "ket":
        ev = state.conj() @ A @ state
    elif stype == "density matrix":
        ev = np.trace(state @ A)
    else:
        raise ValueError("Unknown state type.")
    return ev  # This once had .real, I have removed this now. Aug 7 3:25 pm.


def covariance(D, state, A, B):
    """TODO: switch to other form since catastrophic numerical cancellation can occur here.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""
    return EV(D, state, dag(A) @ B) - EV(D, state, dag(A)) * EV(D, state, B)


def variance(D, state, A):
    return covariance(
        D, state, A, A
    ).real  # This is allowed to have .real since it must be real even for A non-Hermitian.


# operator tests


def supremum_matrix_distance(A, B):
    return np.abs(A - B).max()


def are_equal(A, B, ε=1e-15):
    return supremum_matrix_distance(A, B) <= ε


def is_hermitian(A, ε=1e-15):
    return are_equal(A, dag(A), ε)


def is_unitary(D, A, ε=1e-15):
    I = np.identity(D)
    return are_equal(A @ dag(A), I, ε) & are_equal(dag(A) @ A, I, ε)


# ancilla
def RDM(D, state):
    """Reduced density matrix (RDM) tracing over the second site, i.e. the ancilla."""
    ρ = ρ_from_state(D**2, state)
    Jz_eigvs = np.linalg.eigh(Jz(D)).eigenvectors
    ρs = np.zeros((D, D), dtype=complex)
    for eigv in Jz_eigvs.T:
        # partial trace = sum over the eigenbasis of the ancilla
        eigv_ext = np.kron(I(D), eigv)
        ρs += dag(eigv_ext).T @ ρ @ eigv_ext.T
    return ρs


def entanglement_entropy(D, state, normalised=False):
    """state should be D**2 ket or DM, second site traced out (doesn't matter which site is traced out)"""
    return entropy(RDM(D, state), D, normalised=normalised)


def purify(D, ρ, randomise=True, seed=None, please_print=True):
    """Purify state using GNS construction, with random unitary then acting on the ancilla."""
    λs, vs = np.linalg.eigh(ρ)
    # print(λs, vs, ρ)
    # rounding off small negative eigenvalues since all should be non-negative
    tol = 1e-8
    if np.any(λs < -tol):
        if please_print:
            print(
                f"Eigenvalues of ρ are too negative, e.g. {np.min(λs):.4g} beyond tolerance of {tol:.1g}. Returning NaN."
            )
        return np.nan
    λs[λs < 0] = 0
    # joint ket = \sum_i \sqrt(p_i) |e_i,e_i>
    ket = sum(np.sqrt(λs[i]) * np.kron(vs[:, i], vs[:, i]) for i in range(D))
    # pure DM = \sum_ij \sqrt(p_ip_j) |e_i,e_i><e_j,e_j|
    # DM = sum(np.sqrt(λs[i] * λs[j]) * np.outer(np.kron(vs[:, i], vs[:, i]), np.conjugate(np.kron(vs[:, j], vs[:, j]))) for i in range(D) for j in range(D))
    if not randomise:
        return ket
    else:
        # purification is non-unique up to unitaries on the ancilla
        U = np.kron(I(D), random_unitary(D, seed))
        return U @ ket


# classical methods
def random_real_vector(D, seed=None):
    """Vector in (-1, 1)^D"""
    if seed is None:
        return 2 * np.random.rand(D) - 1
    else:
        rng = np.random.default_rng(seed)
        return 2 * rng.random(D) - 1


# # TODO: refactor these Gram-Schmidt methods, 3 different families of them
# def orthogonal_component_real_vectors(A, B):
#     return A - np.dot(A, B) / np.dot(B, B) * B

# def orthogonal_component_real_vectors_from_set(A, S):
#     Aperp = A
#     for i, B in enumerate(S):
#         Bperp = orthogonal_component_real_vectors_from_set(B, S[:i])
#         Aperp = orthogonal_component_real_vectors(Aperp, Bperp)
#     return Aperp

# def orthogonal_component_real_vectors_from_orthogonal_set(A, S):
#     Aperp = A
#     for B in S:
#         Aperp = orthogonal_component_real_vectors(Aperp, B)
#     return Aperp

# def orthogonalise_set_of_real_vectors(S):
#     S_orth = []
#     for A in S:
#         Aperp = orthogonal_component_real_vectors_from_orthogonal_set(A, S_orth)
#         tol = 1e-15
#         if np.all(np.abs(Aperp) < tol):
#             continue
#         else:
#             S_orth.append(Aperp)
#     return np.array(S_orth) if type(S) is np.ndarray else S_orth


def random_orthogonal_real_vectors(D, num_vecs, seed, **kwargs):
    """Orthogonalisation means that the vectors may no longer be within (-1, 1). To rescale, include normalisation_fn in kwargs."""
    seed_sequence = generate_seed_sequence(seed, length=num_vecs)
    # generate random vectors
    vs = [random_real_vector(D, seed=seed2) for seed2 in seed_sequence]
    # orthogonalise these vectors from eachother and the identity
    # return orthogonalise_set_of_real_vectors([np.ones(D), *vs])[1:]
    return orthogonalise_set_general([np.ones(D), *vs], "real vector", **kwargs)[1:]


def covariance_classical(x, y, p):
    return sum(x.conj() * y * p) - sum(x.conj() * p) * sum(y * p)


def variance_classical(x, p):
    return covariance_classical(x, x, p).real


def hellinger_distance(p1, p2):
    """https://en.wikipedia.org/wiki/Hellinger_distance#Discrete_distributions"""
    return np.linalg.norm(np.sqrt(p1) - np.sqrt(p2)) / np.sqrt(2).real


# plotting style
def lim(axis_data):
    return (axis_data.min(), axis_data.max())


def grid(ax, zorder=0, **kwargs):
    ax.grid("both", "both", color="gainsboro", zorder=zorder, **kwargs)


def legend(ax, reversed=False, **kwargs):
    defaults = dict(handlelength=1, labelspacing=0, frameon=False, handletextpad=0.3)
    if reversed:
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=reversed(h), labels=reversed(l), **(defaults | kwargs))
    else:
        ax.legend(**(defaults | kwargs))


def colorbar(fig, ax, image, pad=0.05, fmt=None, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    # TODO: fix formatter to handle overall order scaling
    # e.g. fmt = '%.1g'
    if fmt is not None:
        fig.colorbar(image, cax=cax, format=tkr.FormatStrFormatter(fmt), **kwargs)
    else:
        fig.colorbar(image, cax=cax, **kwargs)


# visualisation


def Husimi_Q(D, state, plot_points, progress_bar=True):
    """TODO: Haine18 Eq.S23 includes a prefactor that scales with D here, should I?"""
    θs = np.linspace(0, π, plot_points)
    ϕs = np.linspace(0, 2 * π, plot_points)
    Qs = np.zeros((plot_points, plot_points))
    if progress_bar:
        enum = enumerate(tqdm(θs, desc="Calculating Husimi Q-function"))
    else:
        enum = enumerate(θs)
    # for j, θ in enumerate(tqdm(θs, desc='Calculating Husimi Q-function')):
    for j, θ in enum:
        for k, ϕ in enumerate(ϕs):
            CSS_ket = CSS(D, θ, ϕ)
            # 1/π here for normalisation, https://en.wikipedia.org/wiki/Husimi_Q_representation
            if state_type(D, state) == "ket":
                Qs[j, k] = np.abs(CSS_ket.conj() @ state) ** 2 / π
            elif state_type(D, state) == "density matrix":
                Qs[j, k] = (dag(CSS_ket) @ state @ CSS_ket).real / π
            else:
                raise ValueError("Unknown state type.")
    return Qs


def plot_Husimi_Q(
    D,
    state,
    plot_points,
    progress_bar=False,
    return_dont_show=False,
    fig=None,
    ax=None,
    **ax_set_kwargs,
):
    Qs = Husimi_Q(D, state, plot_points, progress_bar=progress_bar)

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(Qs, origin="upper")
    # cbar_label = r"Husimi $Q$-function, $|\langle\theta, \phi|\psi\rangle|^2$"
    cbar_label = r"Husimi $Q$-function"
    colorbar(fig, ax, im, label=cbar_label)
    ticks = [
        0,
        (plot_points - 1) // 4,
        (plot_points - 1) // 2,
        3 * (plot_points - 1) // 4,
        plot_points - 1,
    ]
    θtick_labels = [0, r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    ϕtick_labels = [0, r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2 \pi$"]
    ax.set_xticks(ticks, ϕtick_labels)
    ax.set_yticks(ticks, θtick_labels)
    ax.set(xlabel=r"$\phi$", ylabel=r"$\theta$", **ax_set_kwargs)
    if return_dont_show:
        return fig, ax
    else:
        plt.show()
