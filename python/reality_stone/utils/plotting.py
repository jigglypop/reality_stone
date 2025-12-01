import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def plot_energy_history(history: Dict[str, List[float]], save_path: str = 'energy_history.png'):
    """
    Plots the evolution of Lagrangian, Kinetic, and Potential energies.
    
    Args:
        history: Dictionary containing lists for 'lagrangian', 'kinetic', 'potential'.
        save_path: Path to save the plot.
    """
    if not history:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Check keys exist
    if 'lagrangian' in history:
        axes[0].plot(history['lagrangian'], 'g-o')
        axes[0].set_title('Lagrangian Evolution')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('L = T - V')
        axes[0].grid(True, alpha=0.3)
    
    if 'kinetic' in history:
        axes[1].plot(history['kinetic'], 'b-o')
        axes[1].set_title('Kinetic Energy Evolution')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('T')
        axes[1].grid(True, alpha=0.3)
    
    if 'potential' in history:
        axes[2].plot(history['potential'], 'r-o')
        axes[2].set_title('Potential Energy Evolution')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('V')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   -> Saved energy history to '{save_path}'")
    plt.close(fig)

def plot_distance_distribution(intra_class: List[float], inter_class: List[float], save_path: str = 'distance_distribution.png'):
    """
    Plots histograms of intra-class and inter-class distances.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(intra_class, bins=30, alpha=0.6, label='Intra-class', color='blue')
    plt.hist(inter_class, bins=30, alpha=0.6, label='Inter-class', color='red')
    plt.xlabel('Geodesic Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution: Intra-class vs Inter-class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   -> Saved distance distribution to '{save_path}'")
    plt.close()

def plot_embeddings_2d(embeddings: np.ndarray, labels: np.ndarray, save_path: str = 'embeddings_2d.png'):
    """
    Plots 2D PCA projection of embeddings.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not installed, skipping PCA plot")
        return

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Learned Embeddings (2D PCA projection)')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   -> Saved embeddings visualization to '{save_path}'")
    plt.close()

