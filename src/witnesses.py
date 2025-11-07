"""
Entanglement witness operators and evaluation.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator


def construct_bell_witness():
    """
    Construct entanglement witness for Bell states.
    W = I/2 - |Phi+><Phi+|

    Tr(W * rho) < 0 implies rho is entangled.
    """
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    projector = np.outer(phi_plus, phi_plus.conj())
    witness = np.eye(4) / 2 - projector
    return witness


def construct_ghz_witness(n_qubits):
    """
    Construct entanglement witness for GHZ states.
    W = (n-1)/n * I - |GHZ><GHZ|
    """
    dim = 2 ** n_qubits
    ghz = np.zeros(dim)
    ghz[0] = 1 / np.sqrt(2)
    ghz[-1] = 1 / np.sqrt(2)

    projector = np.outer(ghz, ghz.conj())
    witness = (n_qubits - 1) / n_qubits * np.eye(dim) - projector
    return witness


def evaluate_witness(witness, rho):
    """
    Evaluate entanglement witness on a density matrix.
    Returns Tr(W * rho). Negative value certifies entanglement.
    """
    return np.real(np.trace(witness @ rho))


def measure_witness_circuit(state_circuit, witness, shots=8192, seed=42):
    """
    Estimate witness expectation value from circuit measurements.
    Decomposes the witness into Pauli terms and measures each.

    For simplicity, uses statevector simulation to get the density matrix.
    """
    from qiskit_aer import StatevectorSimulator

    backend = StatevectorSimulator()
    qc = state_circuit.copy()
    qc.save_statevector()
    job = backend.run(qc)
    statevector = job.result().get_statevector()

    state = np.array(statevector)
    rho = np.outer(state, state.conj())

    return evaluate_witness(witness, rho)


def sweep_noise_witness(state_circuit, witness, noise_levels,
                         shots=8192, seed=42):
    """
    Evaluate witness under depolarizing noise at different levels.

    Returns:
        dict with 'noise_levels', 'witness_values'
    """
    from qiskit_aer import StatevectorSimulator

    backend = StatevectorSimulator()
    qc = state_circuit.copy()
    qc.save_statevector()
    job = backend.run(qc)
    statevector = job.result().get_statevector()

    state = np.array(statevector)
    rho_pure = np.outer(state, state.conj())
    dim = rho_pure.shape[0]

    witness_values = []
    for p in noise_levels:
        # depolarizing channel: rho -> (1-p)*rho + p*I/d
        rho_noisy = (1 - p) * rho_pure + p * np.eye(dim) / dim
        val = evaluate_witness(witness, rho_noisy)
        witness_values.append(val)

    return {
        'noise_levels': list(noise_levels),
        'witness_values': witness_values,
    }
