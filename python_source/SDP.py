from spins_symmetric_subspace import *
import cvxpy as cp


def ECQFI_by_SDP_general(
    D,
    K_list,
    Kdot_list,
    QFI_fn,
    t,
    please_print=True,
    round_to=3,
    debug=False,
    add_more=0,
):
    """Calculate the ECQFI via SDP from Kolodynski's thesis Appendix I for the general case.
    debug only works if please_print is also True and then provides more detailed print outs.
    """
    if please_print:
        print("------------------------------")

    num_kraus = len(K_list)
    # gauge matrix for Kraus representations
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)
    # Kdot_tilde = Kdot - i h K
    Kdot_tilde_list = [
        [Kdot_list[i][0] - sum(1j * h[i, j] * K_list[j][0] for j in range(num_kraus))]
        for i in range(num_kraus)
    ]

    K = np.block(K_list)
    Kdot = np.block(Kdot_list)
    # cp.bmat is like np.block but for unknowns
    Kdot_tilde = cp.bmat(Kdot_tilde_list)

    # real scalar such that λ ** 2 = ||Kdot_tilde^\dag Kdot_tilde||
    λ = cp.Variable(pos=True)
    objective = 4 * cp.Minimize(λ**2)

    A = cp.bmat(
        [
            [λ * I(D), dag(Kdot_tilde)],
            [Kdot_tilde, λ * I(num_kraus * D)],
        ]
    )
    constraints = [
        A >> 0,
    ]

    # Form and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if please_print and debug:
        print("Status:", problem.status)

    if problem.status != "optimal":
        raise ValueError("cvxpy failed.")

    ECQFI = problem.value / (4 * t)
    α = dag(Kdot_tilde.value) @ Kdot_tilde.value
    α_norm = max(np.linalg.eigh(α).eigenvalues)
    if please_print:
        print(f"ECQFI / (4 * t): {ECQFI:.4g}")
        if debug:
            print(f"||\\alpha|| / t: {α_norm / t:.4g}")
            # (α_norm / t) / ECQFI = (α_norm / t) / [problem.value / (4 * t)] = 4 α_norm / problem.value
            print(f"4 ||\\alpha|| / ECQFI: {(α_norm / t) / ECQFI:.4g}")

    #######################
    # optimal initial state
    #######################
    # system of equations, M v_ρ = c, where v_ρ is the RDM ρ vectorised
    M = []
    c = []

    # normalisation constraint
    M.append(matrix_vectorise(I(D)))
    c.append(1)
    if please_print and debug:
        print(f"1 constraints from normalisation (makes the system inhomogenous).")

    # eigenspace constraints
    # # TODO: adaptively choose how much to round off
    # round_tos = range(1, 6)
    # [len(maximum_eigenstates(results['α'], round_to=x)) for x in round_tos]
    vs = maximum_eigenstates(α, round_to=round_to, add_more=add_more)
    # projection onto eigenspace,
    Π = sum(ρ_from_ket(v) for v in vs)
    if please_print:
        print(
            f"Eigenspace is {len(vs)}-dimensional after rounding off to {round_to} digits."
        )

    # Π ρ Π = ρ
    # ρ_jk - sum_lm Π_jl ρ_lm Π_mk = 0
    proj_constraints = []
    for j in range(D):
        for k in range(D):
            proj_constraint = np.zeros((D, D), dtype=complex)
            proj_constraint[j, k] = 1
            # ρ_jk
            proj_constraint = matrix_vectorise(proj_constraint)

            x = np.zeros((D, D), dtype=complex)
            for l in range(D):
                for m in range(D):
                    x[l, m] = Π[j, l] * Π[m, k]
            # ρ_jk - sum_lm (Π_jl Π_mk) ρ_lm
            proj_constraint -= matrix_vectorise(x)
            proj_constraints.append(proj_constraint)

    for row in proj_constraints:
        M.append(row)
        c.append(0)
    if please_print and debug:
        print(f"{len(proj_constraints)} = D^2 constraints from Π ρ Π = ρ.")

    # block constraints from Eq.F6
    Kdot_tilde_list = [[Kdot_tilde_list[i][0].value] for i in range(num_kraus)]
    K_list = K_list

    # tr(rho B) = 0
    # sum_jk rho_jk B_kj = 0
    # B = Kdag h' Ktildedot - Ktildedotdag h' K
    # sum(h[i, j] * dag(K_list[i][0]) @ Kdot_tilde_list[j][0] - h[i, j] * dag(Kdot_tilde_list[i][0]) @ K_list[j][0]
    # for i in range(num_kraus) for j in range(num_kraus))
    blocks = []
    for i in range(num_kraus):
        for j in range(num_kraus):
            block = (
                dag(K_list[i][0]) @ Kdot_tilde_list[j][0]
                - dag(Kdot_tilde_list[i][0]) @ K_list[j][0]
            )
            blocks.append(block)
    blocks = np.array(blocks)

    for block in blocks:
        M.append(matrix_vectorise(block.T))
        c.append(0)
    if please_print and debug:
        print(f"{len(blocks)} constraints from Eq.F6.")

    # normalisation + eigenspace + Eq.F6 block constraints
    M = np.array(M)
    c = np.array(c)
    if please_print and debug:
        print(
            f"Solve M v_ρ = c, where M is {M.shape}, v_ρ is {(D ** 2,)} since ρ is {(D, D)}, c is {c.shape}"
        )
        print(
            f"Condition number (wrt Frobenius/trace norm) is {np.linalg.norm(M) * np.linalg.norm(np.linalg.pinv(M)):.1g}"
        )

    # Want to solve A x = b for A_mn x_n = b_m with m > n (over-constrainted)
    # Solution exists if (1_m - A A^-1) b = 0 for (1_m - A_mn A^-1_nm) b_m = 0
    tol = 1e-5
    if please_print and debug:
        print(f"Computing pseudoinverse...")
    err_sol_exists = (I(len(M)) - M @ np.linalg.pinv(M)) @ c
    sol_exists = np.linalg.norm(err_sol_exists) < tol
    if please_print and debug:
        print(
            f"Solution/s exist up to error (Euclidean norm): {np.linalg.norm(err_sol_exists):.1g}"
        )

    output = dict(
        problem=problem,
        λ=λ,
        h=h,
        A=A,
        K=K,
        K_list=K_list,
        Kdot=Kdot,
        Kdot_list=Kdot_list,
        Kdot_tilde=Kdot_tilde,
        Kdot_tilde_list=Kdot_tilde_list,
        α=α,
        α_norm=α_norm,
        ECQFI=ECQFI,
        sol_exists=sol_exists,
    )

    if not sol_exists:
        # raise ValueError('No solution found.')
        # pressing on and tagging when the solution does not exist
        if please_print:
            print("No solution found.")
        return output

    # Solution unique if 1_n - A^-1 A = 0
    err_sol_unique = I(D**2) - np.linalg.pinv(M) @ M
    sol_unique = np.linalg.norm(err_sol_unique, "fro") < tol
    if please_print:
        if sol_unique:
            print("Solution unique.")
        else:
            print("Solution not unique.")

        if debug:
            print(
                f'Solution unique up to error (Frobenius/trace norm): {np.linalg.norm(err_sol_unique, "fro"):.1g}'
            )
            print(
                f"Solution unique up to error (operator/p=2 norm): {np.linalg.norm(err_sol_unique, 2):.1g}"
            )

    # w = 0 solution
    ρ_wZero = matrix_unvectorise(np.linalg.pinv(M) @ c)
    # if debug: print('w=0 RDM: ', ρ_wZero)
    Fq_opt = QFI_fn(ρ_wZero)
    Fq_opt_ratio = Fq_opt / ECQFI
    S_wZero = entropy(D=D, ρ=ρ_wZero, normalised=True)
    if please_print:
        print(f"Optimal state (w = 0): QFI / ECQFI = {Fq_opt_ratio:.4g}")
        tol = 1e-5
        if abs(S_wZero) < tol:
            print("Optimal state (w = 0) is pure.")
        elif abs(1 - S_wZero) < tol:
            print("Optimal state (w = 0) is maximally mixed.")
        else:
            print(f"Optimal state (w = 0): Entropy / max entropy = {S_wZero:.4g}")

    output = output | dict(
        sol_unique=sol_unique,
        ρ_wZero=ρ_wZero,
        Fq_opt_ratio=Fq_opt_ratio,
        S_wZero=S_wZero,
    )

    if not sol_unique:
        # w != 0 solutions
        def generate_non_unique_solution(seed):
            # Solution given by $A^{-1}b + (1 - A^-1 A)w$ for arbitrary $w$.
            # Dimensions: A^{-1}_nm b_m + (1_n - A^{-1}_nm A_mn) w_n
            w = random_ket(D**2, seed=seed)
            ρ_wNonZero = matrix_unvectorise(np.linalg.pinv(M) @ c + err_sol_unique @ w)
            Fq_wNonZero = QFI_fn(ρ_wNonZero)
            Fq_wNonZero_ratio = Fq_wNonZero / ECQFI
            if please_print:
                print(
                    f"Optimal state (w != 0, random): QFI / ECQFI = {Fq_wNonZero_ratio:.4g}"
                )
            output = dict(
                ρ_wNonZero=ρ_wNonZero,
                Fq_wNonZero_ratio=Fq_wNonZero_ratio,
            )
            if Fq_wNonZero is not np.nan:
                S_wNonZero = entropy(D=D, ρ=ρ_wNonZero, normalised=True)
                fidelity_wZero_wNonZero = fidelity(D, ρ_wZero, ρ_wNonZero)
                if please_print:
                    print(
                        f"Optimal state (w != 0, random): Entropy / max entropy = {S_wNonZero:.4g}"
                    )
                    print(
                        f"Fidelity between optimal state (w = 0) and optimal state (w != 0, random): {fidelity_wZero_wNonZero:.4g}"
                    )
                output = output | dict(
                    S_wNonZero=S_wNonZero,
                    fidelity_wZero_wNonZero=fidelity_wZero_wNonZero,
                )
            return output

        # give one example random solution
        results_wNonZero = generate_non_unique_solution(seed=0)
        output = (
            output
            | results_wNonZero
            | dict(generate_non_unique_solution=generate_non_unique_solution)
        )

    return output
