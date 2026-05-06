from spins_symmetric_subspace import *
from QFImixed_finite import QFImixed_finite


def reject_eigenvalues(λ_j, λ_k, tol=1e-15, chop=True):
    """Returns whether the evals should be rejected and the loop continue past them."""
    # drop imaginary values
    if abs(λ_j.imag) > tol or abs(λ_k.imag) > tol:
        raise ValueError(f"Eigenvalues are complex: λ_j={λ_j}, λ_k={λ_k}.")
    else:
        λ_j = λ_j.real
        λ_k = λ_k.real

    return (
        # drop singular values
        (λ_k + λ_j == 0)
        # drop numerically singular values
        or (chop and abs(λ_k.real + λ_j.real) < tol)
        # drop negative values
        or (λ_j.real < 0 or λ_k.real < 0)
    )


def final_state_and_derivative_via_FDM(
    D,
    state,
    channel,
    parameter,
    parameter_step_size,
    return_ρ1_ρ2_too=False,
):
    """Return the (midpoint) state after the channel and its derivative calculated via finite difference.

    This function was called <finite_difference_ρ_and_ρ_dot> previously.
    """
    # initial state
    ρ0 = ρ_from_state(D, state)

    parameter_minus = parameter - parameter_step_size / 2
    parameter_plus = parameter + parameter_step_size / 2
    ρ1 = channel(ρ0, parameter_minus)
    ρ2 = channel(ρ0, parameter_plus)
    # midpoint of the final states
    ρ = normalise(D, (ρ1 + ρ2) / 2)

    # derivative at midpoint
    δρ = ρ2 - ρ1
    ρ_dot = δρ / parameter_step_size
    if not return_ρ1_ρ2_too:
        return ρ, ρ_dot
    else:
        return ρ, ρ_dot, ρ1, ρ2


def sld(
    D, state, channel, parameter, parameter_step_size, chop=True, return_dict=False
):
    if not chop:
        print("Are you sure that you do not want to chop the SLD?")

    args = (D, state, channel, parameter, parameter_step_size)
    if not return_dict:
        ρ, ρ_dot = final_state_and_derivative_via_FDM(*args)
    else:
        ρ, ρ_dot, ρ1, ρ2 = final_state_and_derivative_via_FDM(
            *args, return_ρ1_ρ2_too=True
        )

    # The normalized (unit “length”) eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    evals, evecs = np.linalg.eigh(ρ)

    SLD = np.zeros_like(ρ)
    if return_dict:
        SLD_components = np.zeros_like(ρ)

    # TODO: vectorise this
    for j, (λ_j, ψ_j) in enumerate(zip(evals, evecs.T)):
        for k, (λ_k, ψ_k) in enumerate(zip(evals, evecs.T)):

            if reject_eigenvalues(λ_j, λ_k, chop=chop):
                continue

            # Paris09, Eq 12 or DnD+15, Eq 78
            component = 2 * (np.conj(ψ_k) @ ρ_dot @ ψ_j) / (λ_k + λ_j)
            SLD += component * np.outer(ψ_k, np.conj(ψ_j))
            if return_dict:
                SLD_components[j, k] = component

    if not return_dict:
        return SLD
    else:
        return dict(
            sld=SLD,
            sld_components=SLD_components,
            ρ=ρ,
            ρ_dot=ρ_dot,
            evals=evals,
            evecs=evecs,
            ρ1=ρ1,
            ρ2=ρ2,
        )


def QFI_from_sld(D, state, channel, parameter, parameter_step_size):
    args = (D, state, channel, parameter, parameter_step_size)
    sld_dict = sld(*args, return_dict=True)
    ρ = sld_dict["ρ"]
    L = sld_dict["sld"]
    return np.trace(ρ @ power(L, 2)).real


def QFI_from_deriv(ρ, ρ_dot):
    # ρ_dot is derivative (called deriv in sld()) calculated from Λ_dot or finite difference method
    # TODO: use the positive definiteness of rho to find them faster
    evals, evecs = np.linalg.eigh(ρ)

    FQ_element = np.zeros_like(ρ)

    # TODO: vectorise this, but respect the reject_eigevalues and nansum. There's probably code out there that already calculates FI fast.
    for j, (λ_j, ψ_j) in enumerate(zip(evals, evecs.T)):
        for k, (λ_k, ψ_k) in enumerate(zip(evals, evecs.T)):
            if reject_eigenvalues(λ_j, λ_k):
                continue

            # from Simon's QFImixed_finite.m
            FQ_element[j, k] = (
                4 * λ_j * np.abs(np.conj(ψ_k) @ ρ_dot @ ψ_j) ** 2 / (λ_k + λ_j) ** 2
            )

    FQ = np.nansum(FQ_element).real
    return FQ


def QFI_finite_difference(D, state, channel, parameter, parameter_step_size):
    """Return the QFI calculated via the finite difference method. Should be the same as QFI_from_sld."""
    args = (D, state, channel, parameter, parameter_step_size)
    ρ, ρ_dot = final_state_and_derivative_via_FDM(*args)
    return QFI_from_deriv(ρ, ρ_dot)


def CFI(probs1, probs2, parameter_step_size, debug=False):
    """Return CFI calculated via finite difference.

    Args:
        probs1: probability distribution
        probs2: infinitesimally different probability distribution
        parameter_step_size (float): step size for finite difference method
    """
    probs_mid = 0.5 * (probs1 + probs2)
    deriv = (probs2 - probs1) / parameter_step_size
    cfi_summand = deriv**2 / probs_mid
    Fc = np.nansum(cfi_summand)
    if not debug:
        return Fc
    else:
        return Fc, cfi_summand


def QFI_FDM_Simon_from_ρ1_ρ2(ρ1, ρ2, parameter_step_size):
    """Return QFI calculated via finite difference.

    Args:
        ρ1: density matrix
        ρ2: infinitesimally different density matrix
        parameter_step_size (float): step size for finite difference method
    """
    Fq = 1 / parameter_step_size**2 * QFImixed_finite(ρ1, ρ2).real
    return Fq


def QFI_FDM_Simon(D, state, channel, parameter, parameter_step_size):
    args = (D, state, channel, parameter, parameter_step_size)
    ρ, ρ_dot, ρ1, ρ2 = final_state_and_derivative_via_FDM(*args, return_ρ1_ρ2_too=True)
    return QFI_FDM_Simon_from_ρ1_ρ2(ρ1, ρ2, parameter_step_size)


def CFI_from_measuring_SLD(
    D,
    state,
    channel,
    parameter,
    parameter_step_size,
    debug=False,
):
    """Returns the CFI wrt the basis of the SLD for the given state."""
    args = (D, state, channel, parameter, parameter_step_size)
    SLD_dict = sld(*args, return_dict=True)
    SLD = SLD_dict["sld"]
    ρ1 = SLD_dict["ρ1"]
    ρ2 = SLD_dict["ρ2"]

    # TODO: check that the evecs are orthogonal since SLD should be Hermitian (check that too)
    # TODO: check that the evals are nondegenerate such that the measurements can be distinguished
    _, evecs = np.linalg.eigh(SLD)

    probs1 = np.zeros(D)
    probs2 = np.zeros(D)
    # TODO: use matrix algebra to calculate this faster
    for j in range(D):
        ψ_j = evecs[:, j]
        prob1 = np.conj(ψ_j) @ ρ1 @ ψ_j
        prob2 = np.conj(ψ_j) @ ρ2 @ ψ_j

        tol = 1e-15
        if abs(prob1.imag) > tol or abs(prob2.imag) > tol:
            raise ValueError("Probabilities are complex.")
        else:
            prob1 = prob1.real
            prob2 = prob2.real

        probs1[j] = prob1
        probs2[j] = prob2

    Fc = CFI(probs1, probs2, parameter_step_size)
    if not debug:
        return Fc
    else:
        return Fc, probs1, probs2
