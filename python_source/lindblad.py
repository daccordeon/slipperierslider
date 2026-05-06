from spins_symmetric_subspace import *
from SDP import ECQFI_by_SDP_general


# analytic QFI
def QFI_noiseless_Lindblad_channel(D, ket, L1, t):
    """Analytic result in the limit of short time and zero signal."""
    varL1 = variance(D, ket, L1)
    return 4 * t * varL1


def QFI_simultaneous_Lindblad_channel_no_prefactor(D, ket, L1, L2, tol=1e-10):
    varL1 = variance(D, ket, L1)
    varL2 = variance(D, ket, L2)
    covL1L2 = covariance(D, ket, L1, L2)
    if np.abs(covL1L2) > tol:
        if varL2 < 0:
            print(
                f"varL2 is negative: {varL2:.3g} and |covL1L2| = {np.abs(covL1L2):.3g} (which should be zero) is not beneath tol = {tol:.1g}. Returning varL1 = {varL1:.3g} instead."
            )
            return varL1
        return varL1 - np.abs(covL1L2) ** 2 / varL2
    else:
        return varL1


def QFI_simultaneous_Lindblad_channel(D, ket, L1, L2, t, tol=1e-10):
    """Analytic result in the limit of short time and zero signal. Note that it is independent of γ2."""
    c11AbsSqr = QFI_simultaneous_Lindblad_channel_no_prefactor(D, ket, L1, L2, tol=tol)
    return 4 * t * c11AbsSqr


def QFI_two_noise_operators_no_prefactor(D, ket, Lsignal, Lnoise1, Lnoise2, tol=1e-10):
    L1 = Lsignal
    L2 = Lnoise1
    L3 = Lnoise2
    var1 = variance(D, ket, L1)
    var2 = variance(D, ket, L2)
    var3 = variance(D, ket, L3)
    cov12 = covariance(D, ket, L1, L2)
    cov23 = covariance(D, ket, L2, L3)
    cov31 = covariance(D, ket, L3, L1)
    # Fq = var1
    # if np.abs(cov12) > tol:
    #     Fq += -np.abs(cov12)**2 / var2
    # return Fq
    return (
        var1
        - np.abs(cov12) ** 2 / var2
        - np.abs(cov31 - cov23.conj() * cov12.conj() / var2) ** 2
        / (var3 - np.abs(cov23) ** 2 / var2)
    )


# Lindblad channel
def Lindblad_term(ρ, L):
    return L @ ρ @ dag(L) - anticommutator(dag(L) @ L, ρ) / 2


def noiseless_Lindblad_channel(D, state, L1, γ1, t):
    ρ = ρ_from_state(D, state)
    return ρ + γ1 * t * Lindblad_term(ρ, L1)


def simultaneous_Lindblad_channel(D, state, L1, L2, γ1, γ2, t):
    ρ = ρ_from_state(D, state)
    return ρ + γ1 * t * Lindblad_term(ρ, L1) + γ2 * t * Lindblad_term(ρ, L2)


def derivative_of_Lindblad_channel(ρ, L1, γ1, t):
    """This is independent of γ2 and the same with and without noise."""
    return 2 * np.sqrt(γ1) * t * Lindblad_term(ρ, L1)


# dual map of the Lindblad channel
def Lindblad_term_for_dual_map(A, L):
    """See notes, from finding the Kraus operators of the channel above and then swapping them. K0 is Hermitian such that the NHH is the same."""
    return dag(L) @ A @ L - anticommutator(dag(L) @ L, A) / 2


def dual_map_of_noiseless_Lindblad_channel(A, L1, γ1, t):
    return A + γ1 * t * Lindblad_term_for_dual_map(A, L1)


def dual_map_of_simultaneous_Lindblad_channel(A, L1, L2, γ1, γ2, t):
    return (
        A
        + γ1 * t * Lindblad_term_for_dual_map(A, L1)
        + γ2 * t * Lindblad_term_for_dual_map(A, L2)
    )


def derivative_of_dual_map(A, L1, γ1, t):
    """This is independent of γ2 and the same with and without noise.
    TODO: check that the derivative of the dual equals the dual of the derivative. What does the dual of the derivative mean though, what are its Kraus operators since it is not a channel? Do they even exist?
    """
    return 2 * np.sqrt(γ1) * t * Lindblad_term_for_dual_map(A, L1)


def QFI_per_4t_any_K_N(
    D, state, Lsigs, Lnoises, please_print=False, **purification_kwargs
):
    """QFI / (4 t) for a ket (unextended or extended) for any K, N. If given an RDM, then uses a purification.
    TODO: Instead of purifying the RDM, try generally solving the recursive formula for the QFI to be just in terms of EVs.
    """
    if state_type(D, state) == "density matrix":
        # extend RDM to a ket, need to also extend the operators
        ket = purify(D, state, please_print=please_print, **purification_kwargs)
        if ket is np.nan:
            return np.nan
        Lnoises = [np.kron(Lnoise, I(D)) for Lnoise in Lnoises]
        Lsigs = [np.kron(Lsig, I(D)) for Lsig in Lsigs]
        D = D**2
    else:
        ket = state
    # images of the noise operators
    images = [ket, *[Lnoise @ ket for Lnoise in Lnoises]]
    # vs = orthogonalise_set_of_kets(D, images, please_normalise=True)
    vs = orthogonalise_set_general(images, "ket", normalisation_fn=True, D=D)
    # else the noises were degenerate for this ket, which is fine (perhaps desirable), but need to then check the QFI definition
    if len(vs) < len(images) and please_print:
        print("Noises are degenerate.")
    # Π = |0><0| + |K+1><K+1| + ... + |K+N><K+N|
    Π = sum(ρ_from_ket(v) for v in vs)
    return sum(EV(D, ket, dag(Lsig) @ (I(D) - Π) @ Lsig) for Lsig in Lsigs).real


def generate_K_Kdot_for_K1_N1(D, L1, L2, t, γ1, γ2):
    # vector of Kraus operators
    # TODO: take the γ1 -> 0 limit here explicitly in the Kraus operators and derivatives
    K_list = [
        [I(D) - 0.5 * t * (γ1 * dag(L1) @ L1 + γ2 * dag(L2) @ L2)],
        [np.sqrt(γ1 * t) * L1],
        [np.sqrt(γ2 * t) * L2],
    ]
    # derivative wrt sqrt(γ1)
    Kdot_list = [
        [-t * np.sqrt(γ1) * dag(L1) @ L1],
        [np.sqrt(t) * L1],
        [np.zeros((D, D))],
    ]
    return dict(K_list=K_list, Kdot_list=Kdot_list)


def generate_K_Kdot_for_any_K_N(D, Lsigs, Lnoises, t, γ1, γnoises):
    # vector of Kraus operators
    # TODO: take the γ1 -> 0 and small t approximation limit here explicitly in the Kraus operators and derivatives
    K = len(Lsigs)
    N = len(Lnoises)
    # if given one number, then take it as uniform but uncorrelated across all noises
    if type(γnoises) is float or type(γnoises) is int:
        γnoises = [γnoises for _ in range(N)]
    # assert N == len(γnoises)
    Ls = [*Lsigs, *Lnoises]
    γs = [*[γ1 for _ in Lsigs], *γnoises]
    K_list = [
        [I(D) - 0.5 * t * sum(γ * dag(L) @ L for (L, γ) in zip(Ls, γs))],
        *[[np.sqrt(γ * t) * L] for (L, γ) in zip(Ls, γs)],
    ]
    # derivative wrt sqrt(γ1)
    Kdot_list = [
        [-t * sum(np.sqrt(γ1) * dag(Lsig) @ Lsig for Lsig in Lsigs)],
        *[[np.sqrt(t) * Lsig] for Lsig in Lsigs],
        *[[np.zeros((D, D))] for _ in Lnoises],
    ]
    return dict(K_list=K_list, Kdot_list=Kdot_list)


def ECQFI_by_SDP_for_K1_N1(
    D, L1, L2, t, γ1, γ2, please_print=True, round_to=3, debug=False
):
    """Calculate the ECQFI via SDP from Kolodynski's thesis Appendix I
    for the estimation problem of signal operator L1 in the presence of noise operator L2.
    debug only works if please_print is also True and then provides more detailed print outs.
    """
    kraus = generate_K_Kdot_for_K1_N1(D, L1, L2, t, γ1, γ2)
    QFI_fn = lambda ρ: QFI_simultaneous_Lindblad_channel_no_prefactor(D, ρ, L1, L2)

    return ECQFI_by_SDP_general(
        D,
        kraus["K_list"],
        kraus["Kdot_list"],
        QFI_fn,
        t,
        please_print=please_print,
        round_to=round_to,
        debug=debug,
    )


def ECQFI_by_SDP_for_any_K_N(
    D,
    Lsigs,
    Lnoises,
    t,
    γ1,
    γnoises,
    please_print=True,
    round_to=3,
    debug=False,
    add_more=0,
    **purification_kwargs,
):
    """Calculate the ECQFI via SDP from Kolodynski's thesis Appendix I
    for the estimation problem of signal operator L1 in the presence of noise operator L2.
    debug only works if please_print is also True and then provides more detailed print outs.
    """
    kraus = generate_K_Kdot_for_any_K_N(D, Lsigs, Lnoises, t, γ1, γnoises)
    QFI_fn = lambda ρ: QFI_per_4t_any_K_N(
        D, ρ, Lsigs, Lnoises, please_print=please_print, **purification_kwargs
    )
    return ECQFI_by_SDP_general(
        D,
        kraus["K_list"],
        kraus["Kdot_list"],
        QFI_fn,
        t,
        please_print=please_print,
        round_to=round_to,
        debug=debug,
        add_more=add_more,
    )
