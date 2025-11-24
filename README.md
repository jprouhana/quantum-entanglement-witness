# Quantum Entanglement Detection and Quantification

Tools for detecting and measuring entanglement in multi-qubit quantum states using witness operators, concurrence, and negativity.

## Overview

Entanglement is a key resource in quantum computing and QML. This project implements methods for preparing entangled states (Bell, GHZ, W, cluster), measuring entanglement via computable metrics, and constructing entanglement witness operators that can certify entanglement from measurement data.

## Structure

```
src/
  state_preparation.py  # Bell, GHZ, W, cluster state circuits
  entanglement.py       # Concurrence, negativity, von Neumann entropy
  witnesses.py          # Entanglement witness construction and evaluation
  tomography_utils.py   # Partial trace and density matrix reconstruction
  plotting.py           # Entanglement maps and witness expectation plots
notebooks/
  entanglement_analysis.ipynb  # Full analysis notebook
```

## Key Results

| State   | Qubits | Concurrence | Negativity | Witness <W> |
|---------|--------|-------------|------------|-------------|
| Bell    | 2      | 1.000       | 0.500      | -0.500      |
| GHZ     | 3      | —           | 0.500      | -0.500      |
| W       | 3      | —           | 0.471      | -0.333      |
| Cluster | 4      | —           | 0.500      | -0.250      |

Negative witness values confirm entanglement.

## References

- Horodecki et al., "Quantum entanglement" (2009)
- Guhne & Toth, "Entanglement detection" (2009)
- Wootters, "Entanglement of formation" (1998)

## Requirements

```
pip install -r requirements.txt
```
