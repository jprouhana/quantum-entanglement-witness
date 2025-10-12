"""
Utilities for density matrix reconstruction from statevector simulation.
"""

import numpy as np
from qiskit_aer import StatevectorSimulator


def get_density_matrix(circuit):
    """
    Get the density matrix of a circuit via statevector simulation.

    Args:
        circuit: QuantumCircuit (without measurements)

    Returns:
        density matrix as numpy array
    """
    qc = circuit.copy()
    qc.save_statevector()
    backend = StatevectorSimulator()
    job = backend.run(qc)
    statevector = np.array(job.result().get_statevector())
    return np.outer(statevector, statevector.conj())


def state_fidelity(rho, sigma):
    """
    Compute fidelity between two density matrices.
    F(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2
    """
    from scipy.linalg import sqrtm
    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)
    fidelity = np.real(np.trace(sqrt_product)) ** 2
    return min(1.0, max(0.0, fidelity))


def add_depolarizing_noise(rho, p):
    """Apply depolarizing noise: rho -> (1-p)*rho + p*I/d."""
    dim = rho.shape[0]
    return (1 - p) * rho + p * np.eye(dim) / dim


def purity(rho):
    """Compute purity Tr(rho^2). 1 for pure states, 1/d for maximally mixed."""
    return np.real(np.trace(rho @ rho))
