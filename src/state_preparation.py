"""
Quantum state preparation circuits for entanglement experiments.
"""

from qiskit.circuit import QuantumCircuit


def bell_state(variant='phi_plus'):
    """
    Prepare a two-qubit Bell state.
    Variants: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    if variant == 'phi_minus':
        qc.z(0)
    elif variant == 'psi_plus':
        qc.x(1)
    elif variant == 'psi_minus':
        qc.x(1)
        qc.z(0)

    return qc


def ghz_state(n_qubits):
    """
    Prepare an n-qubit GHZ state: (|00...0> + |11...1>) / sqrt(2)
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def w_state(n_qubits):
    """
    Prepare an n-qubit W state: equal superposition of single-excitation
    basis states (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)
    """
    import numpy as np
    qc = QuantumCircuit(n_qubits)

    # prepare W state using a cascade of controlled rotations
    qc.x(0)
    for i in range(n_qubits - 1):
        theta = 2 * np.arccos(np.sqrt(1 / (n_qubits - i)))
        qc.cry(theta, i, i + 1)
        qc.cx(i + 1, i)

    return qc


def cluster_state(n_qubits):
    """
    Prepare a 1D cluster state: apply H to all qubits, then CZ between neighbors.
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    return qc


def product_state(n_qubits, angles=None):
    """
    Prepare a separable (product) state as a control/baseline.
    Each qubit gets an independent Ry rotation.
    """
    import numpy as np
    qc = QuantumCircuit(n_qubits)
    if angles is None:
        angles = [np.pi / 4] * n_qubits
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    return qc


def get_state_builders():
    """Return dict of state preparation functions."""
    return {
        'Bell': lambda: bell_state('phi_plus'),
        'GHZ-3': lambda: ghz_state(3),
        'W-3': lambda: w_state(3),
        'Cluster-4': lambda: cluster_state(4),
        'Product-2': lambda: product_state(2),
    }
