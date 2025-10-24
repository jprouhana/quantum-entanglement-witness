"""
Entanglement quantification metrics.
"""

import numpy as np
from scipy.linalg import sqrtm


def partial_trace(rho, n_qubits, keep):
    """
    Compute partial trace of a density matrix.

    Args:
        rho: density matrix (2^n x 2^n)
        n_qubits: total number of qubits
        keep: list of qubit indices to keep

    Returns:
        reduced density matrix
    """
    dims = [2] * n_qubits
    rho_tensor = rho.reshape(dims + dims)

    trace_out = [i for i in range(n_qubits) if i not in keep]

    # trace out qubits by contracting indices
    for idx in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + n_qubits)
        n_qubits -= 1
        # adjust remaining indices
        for j in range(len(keep)):
            if keep[j] > idx:
                keep[j] -= 1

    dim_keep = 2 ** len(keep)
    return rho_tensor.reshape(dim_keep, dim_keep)


def concurrence(rho):
    """
    Compute concurrence for a two-qubit density matrix.
    C(rho) = max(0, l1 - l2 - l3 - l4) where li are eigenvalues
    of sqrt(sqrt(rho) * rho_tilde * sqrt(rho)) in decreasing order.
    """
    sy = np.array([[0, -1j], [1j, 0]])
    sigma_yy = np.kron(sy, sy)

    rho_tilde = sigma_yy @ rho.conj() @ sigma_yy
    sqrt_rho = sqrtm(rho)

    R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]

    return max(0, eigenvalues[0] - sum(eigenvalues[1:]))


def negativity(rho, n_qubits, partition=None):
    """
    Compute negativity via partial transpose.
    N(rho) = (||rho^{T_B}||_1 - 1) / 2

    Args:
        rho: density matrix
        n_qubits: total qubits
        partition: qubit index for bipartition (default: first qubit)
    """
    if partition is None:
        partition = 0

    dim = 2 ** n_qubits
    d_a = 2
    d_b = dim // d_a

    # reshape for partial transpose on subsystem B
    rho_reshaped = rho.reshape(d_a, d_b, d_a, d_b)
    rho_pt = rho_reshaped.transpose(0, 3, 2, 1).reshape(dim, dim)

    eigenvalues = np.linalg.eigvalsh(rho_pt)
    neg = (np.sum(np.abs(eigenvalues)) - 1) / 2

    return max(0, neg)


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho log2 rho)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def entanglement_entropy(rho, n_qubits, partition_qubits):
    """
    Compute entanglement entropy for a bipartition.
    S_A = -Tr(rho_A log2 rho_A)
    """
    rho_a = partial_trace(rho, n_qubits, list(partition_qubits))
    return von_neumann_entropy(rho_a)
