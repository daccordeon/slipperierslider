import numpy as np


def QFImixed_finite(rho1, rho2):
    # WARNING! CURRENTLY HAS TROUBLE WITH PURE STATES IN A LARGE HILBERT SPACE
    # I'M NOT SURE WHY.

    # convex sum of density matrices is a valid state still (PO with T1)
    rho = 0.5 * (rho1 + rho2)
    rho = rho / np.trace(rho)

    Nt = len(np.diag(rho)) - 1

    drho = rho2 - rho1

    # The normalized (unit “length”) eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    # D is evals, V is evecs
    # TODO: discard negative evals in D?
    D, V = np.linalg.eigh(rho)
    # print(D, V)

    FQ_element = np.zeros((Nt + 1, Nt + 1))

    for j in range(Nt + 1):
        # breakpoint()
        psi_j = V[:, j]
        lambda_j = D[j]

        for jj in range(Nt + 1):
            psi_jj = V[:, jj]
            lambda_jj = D[jj]

            # drop negative evals like QuTiP, FQ_element[j, jj] remains zero
            if lambda_j.real < 0 or lambda_jj.real < 0:
                continue

            # drop singular values
            if lambda_jj + lambda_j == 0:
                continue

            # breakpoint()
            tolerance = 1e-15
            if abs(lambda_j.imag) > tolerance or abs(lambda_jj.imag) > tolerance:
                print(lambda_j, lambda_jj)
                raise ValueError("Eigenvalues are complex.")
            else:
                # TODO: check that this is not editing the array in-place
                lambda_j = lambda_j.real
                lambda_jj = lambda_jj.real

            # psi_jj, psi_j just scalars, but should be vectors here?
            # (Pdb) np.conj(psi_jj) @ drho @ psi_j
            # *** ValueError: matmul: Input operand 0 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)
            # TODO: understand why this is 4 lambda_j / (lambda_jj + lambda_j) ** 2 rather than 2 (lambda_j - lambda_jj)^2/ (lambda_jj + lambda_j) as in PS09 or PS13-S.5
            # TODO: check that lambda_jj + lambda_j != 0, but these are already filtered out by the NaNs? Check that we don't have a numerical error where they should be dropped

            # from SLD in Paris09 or DnD+18 review Eq 78
            FQ_element[j, jj] = (
                4
                * lambda_j
                * np.abs(np.conj(psi_jj) @ drho @ psi_j) ** 2
                / (lambda_jj + lambda_j) ** 2
            )
            # in QFImixed.m, for a given rho and Jn rather than rho1 and rho2 (so the above must be something else?):
            # FQ_element(j,jj) = 2*abs(psi_jj'*Jn*psi_j)^2*(lambda_jj-lambda_j)^2/(lambda_j+lambda_jj);
            # we do not have an observable (unless the si=0 correspondence with generator x holds), so we cannot use this form

            # RuntimeWarning: invalid value encountered in double_scalars
            # print(
            #     drho,
            #     psi_j,
            #     np.conj(psi_jj),
            #     np.conj(psi_jj) @ drho @ psi_j,
            #     np.abs(np.conj(psi_jj) @ drho @ psi_j) ** 2,
            #     FQ_element[j, jj],
            #     sep="\n",
            # )
            # break

    # largest negative element
    # print(np.min(FQ_element[FQ_element<0]))
    FQ = np.nansum(FQ_element)

    # else:
    # finding pure-state form of rho:
    # THE REASON THIS DOESN'T WORK IS BECAUSE THE GLOBAL PHASE OF THE
    # EIGENVECTOR OF RHO IS SCRAMBLED
    # MAYBE DO THIS WITH FIDELITY INSTEAD

    # V, D = np.linalg.eigh(rho1)
    # permutation = np.argsort(np.diag(D))
    # lambda1 = D[permutation, permutation]
    # psivector = V[:, permutation]
    # psi1 = psivector[:, Nt]

    # V, D = np.linalg.eigh(rho2)
    # permutation = np.argsort(np.diag(D))
    # lambda2 = D[permutation, permutation]
    # psivector = V[:, permutation]
    # psi2 = psivector[:, Nt]

    # psi = 0.5 * (psi1 + psi2)
    # dpsi = (psi2 - psi1)

    # FQ = 4 * (np.conj(dpsi) @ dpsi - np.abs(np.conj(psi) @ dpsi)**2)

    # end
    return FQ
