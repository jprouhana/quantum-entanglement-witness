"""
Visualization for entanglement experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_density_matrix(rho, title='Density Matrix', save_dir='results',
                         filename='density_matrix.png'):
    """Plot real and imaginary parts of a density matrix."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(np.real(rho), cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    axes[0].set_title(f'{title} (Real)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.imag(rho), cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    axes[1].set_title(f'{title} (Imaginary)')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150)
    plt.close()


def plot_entanglement_metrics(state_names, metrics_dict, save_dir='results'):
    """Bar chart comparing entanglement metrics across states."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    n_states = len(state_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_states)
    width = 0.8 / n_metrics
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']

    for i, metric in enumerate(metric_names):
        values = metrics_dict[metric]
        ax.bar(x + i * width, values, width, label=metric,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel('Quantum State', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Entanglement Metrics Comparison')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(state_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path / 'entanglement_metrics.png', dpi=150)
    plt.close()


def plot_witness_vs_noise(noise_levels, witness_values, state_name='',
                           save_dir='results'):
    """Plot witness expectation value as a function of noise."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_levels, witness_values, 'o-', color='#FF6B6B',
            linewidth=2, markersize=6)
    ax.axhline(y=0, color='#2C3E50', linestyle='--', alpha=0.5,
               label='Entanglement boundary')
    ax.fill_between(noise_levels, min(witness_values), 0,
                     alpha=0.1, color='#4ECDC4', label='Entangled region')

    ax.set_xlabel('Depolarizing Noise Rate', fontsize=12)
    ax.set_ylabel('Witness Expectation <W>', fontsize=12)
    ax.set_title(f'Entanglement Witness vs Noise ({state_name})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'witness_vs_noise.png', dpi=150)
    plt.close()
