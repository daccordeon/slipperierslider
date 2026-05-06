import pandas as pd
from spins_symmetric_subspace import *
from fisher_information import *


def next_ket_from_SLD(SLD, Λ_dual, Λ_dot_dual):
    """Return the next ket using the SLD, see Macieszczak13 Eq.11."""
    G_SLD = 2 * Λ_dot_dual(SLD) - Λ_dual(power(SLD, 2))
    return extreme_eigenstate(G_SLD, "max")


def next_ket_from_ket(
    D, ket, channel, parameter, parameter_step_size, Λ_dual, Λ_dot_dual
):
    SLD = sld(D, ket, channel, parameter, parameter_step_size)
    ket_next = next_ket_from_SLD(
        SLD,
        Λ_dual,
        Λ_dot_dual,
    )
    return ket_next


def variational_algorithm(
    D,
    ket0,
    channel,
    parameter,
    parameter_step_size,
    Λ_dual,
    Λ_dot_dual,
    num_iter=2,
    running_prints=False,
    print_every=None,
    run_from_script=False,
    calculate_QFI=True,
    progress_bar=True,
    QFI_fn=None,
    calculate_QFI_numerically_too=False,
):
    """TODO: See old bosonic code for additional functionality"""

    if not run_from_script:
        kets = []
        if calculate_QFI:
            qfis = []
            if QFI_fn is not None and calculate_QFI_numerically_too:
                qfis_numeric = []
        if progress_bar:
            itr = tqdm(range(num_iter + 1), desc="Variational algorithm")
        else:
            itr = range(num_iter + 1)
    else:
        itr = range(num_iter + 1)

    for i in itr:
        if i == 0:
            ket = ket0
        else:
            ket = next_ket_from_ket(
                D,
                kets[-1] if not run_from_script else ket,
                channel,
                parameter,
                parameter_step_size,
                Λ_dual,
                Λ_dot_dual,
            )

        if not run_from_script:
            kets.append(ket)

        if calculate_QFI:
            # results are not recovered by later QFI calls?
            if QFI_fn is None:
                qfi_i = QFI_finite_difference(
                    D,
                    ket,
                    channel,
                    parameter,
                    parameter_step_size,
                )
            else:
                qfi_i = QFI_fn(ket)
                if calculate_QFI_numerically_too:
                    qfis_numeric.append(
                        QFI_finite_difference(
                            D,
                            ket,
                            channel,
                            parameter,
                            parameter_step_size,
                        )
                    )
            qfis.append(qfi_i)

        if running_prints:
            message = (
                f"Iteration {i}: QFI = {qfis[i]:.3f}"
                if calculate_QFI
                else f"Iteration {i}"
            )
            print(message)

        if (
            calculate_QFI
            and print_every is not None
            and (i % print_every == 0 or i == num_iter)
        ):
            message = f"Iteration {i}: QFI = {qfi_i:.3f}, ket = {ket}"
            print(message)

    if not run_from_script:
        df = pd.DataFrame()
        df["itr"] = range(num_iter + 1)
        if calculate_QFI:
            if calculate_QFI_numerically_too:
                df["QFI analytic"] = qfis
                df["QFI numerical"] = qfis_numeric
            else:
                df["QFI"] = qfis
        df["ket"] = kets
        if calculate_QFI:
            if calculate_QFI_numerically_too:
                df = df.sort_values("QFI analytic", ascending=False)
            else:
                df = df.sort_values("QFI", ascending=False)
        return df
