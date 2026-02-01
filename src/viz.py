"""Visualization utilities for STEM-HAADF Crystal Domain Segmentation Pipeline."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Consistent figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10


def save_image_png(image: np.ndarray, path: str, cmap: str = 'gray', 
                   vmin: Optional[float] = None, vmax: Optional[float] = None,
                   title: Optional[str] = None, colorbar: bool = True) -> None:
    """
    Save image as PNG with optional colormap.
    
    Args:
        image: 2D numpy array
        path: Output path for PNG
        cmap: Matplotlib colormap name
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        title: Optional figure title
        colorbar: Whether to include colorbar
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_fft_png(fft_power: np.ndarray, path: str, log_scale: bool = True,
                 title: str = "FFT Power Spectrum") -> None:
    """
    Save FFT power spectrum as PNG.
    
    Args:
        fft_power: 2D FFT power spectrum (already shifted so DC is centered)
        path: Output path for PNG
        log_scale: Apply log scaling for better visualization
        title: Figure title
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if log_scale:
        # Add small epsilon to avoid log(0)
        display_data = np.log1p(fft_power)
        cbar_label = 'Log(Power + 1)'
    else:
        display_data = fft_power
        cbar_label = 'Power'
    
    im = ax.imshow(display_data, cmap='inferno', origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    
    ax.set_title(title)
    ax.set_xlabel('kx (frequency)')
    ax.set_ylabel('ky (frequency)')
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_labeled_image(labels: np.ndarray, path: str, cmap: str = 'tab20',
                       title: str = "Domain Labels") -> None:
    """
    Save domain labels with discrete colormap.
    
    Args:
        labels: 2D integer array of domain labels
        path: Output path for PNG
        cmap: Matplotlib colormap name for discrete labels
        title: Figure title
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Create discrete colormap
    base_cmap = plt.cm.get_cmap(cmap)
    colors = [base_cmap(i % base_cmap.N) for i in range(n_labels)]
    discrete_cmap = ListedColormap(colors)
    
    # Create boundaries for discrete colorbar
    bounds = np.arange(n_labels + 1) - 0.5
    norm = BoundaryNorm(bounds, discrete_cmap.N)
    
    im = ax.imshow(labels, cmap=discrete_cmap, norm=norm, origin='upper')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                        ticks=np.arange(n_labels))
    cbar.set_label('Domain ID')
    cbar.ax.set_yticklabels([str(l) for l in unique_labels])
    
    ax.set_title(f"{title} (n={n_labels})")
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_overlay(image: np.ndarray, labels: np.ndarray, path: str, 
                 alpha: float = 0.3, title: str = "Domain Overlay") -> None:
    """
    Overlay domain boundaries on original image.
    
    Args:
        image: 2D original image
        labels: 2D domain labels (same shape as image or will be resized)
        path: Output path for PNG
        alpha: Transparency for domain overlay
        title: Figure title
    """
    from scipy.ndimage import zoom
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Resize labels to match image if needed
    if labels.shape != image.shape:
        zoom_factors = (image.shape[0] / labels.shape[0], 
                        image.shape[1] / labels.shape[1])
        labels_resized = zoom(labels, zoom_factors, order=0)
    else:
        labels_resized = labels
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Normalize image for display
    img_normalized = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image) + 1e-8)
    
    # Show grayscale image
    ax.imshow(img_normalized, cmap='gray', origin='upper')
    
    # Overlay domains with transparency
    unique_labels = np.unique(labels_resized)
    n_labels = len(unique_labels)
    
    # Create colormap for overlay
    cmap = plt.cm.get_cmap('tab20')
    colors = np.zeros((*labels_resized.shape, 4))
    
    for i, label in enumerate(unique_labels):
        mask = labels_resized == label
        color = cmap(i % cmap.N)
        colors[mask] = color
    
    colors[..., 3] = alpha  # Set transparency
    ax.imshow(colors, origin='upper')
    
    # Draw boundaries
    from scipy.ndimage import sobel
    boundaries = np.zeros_like(labels_resized, dtype=bool)
    for axis in [0, 1]:
        boundaries |= (sobel(labels_resized.astype(float), axis=axis) != 0)
    
    # Overlay boundaries in white
    boundary_overlay = np.zeros((*labels_resized.shape, 4))
    boundary_overlay[boundaries] = [1, 1, 1, 0.8]  # White boundaries
    ax.imshow(boundary_overlay, origin='upper')
    
    ax.set_title(f"{title} ({n_labels} domains)")
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(i % cmap.N), 
                            label=f'Domain {label}')
                       for i, label in enumerate(unique_labels)]
    ax.legend(handles=legend_elements, loc='upper right', 
              fontsize=8, framealpha=0.8)
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_embedding_plot(features: np.ndarray, labels: np.ndarray, path: str,
                        method: str = 'PCA', title: Optional[str] = None) -> None:
    """
    Save 2D embedding scatter plot colored by labels.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Cluster labels for each sample
        path: Output path for PNG
        method: Dimensionality reduction method used (for title)
        title: Optional custom title
    """
    from sklearn.decomposition import PCA
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Reduce to 2D if needed
    if features.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(features)
        var_explained = pca.explained_variance_ratio_.sum() * 100
    else:
        embedding = features
        var_explained = 100.0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20')
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[cmap(i % cmap.N)], label=f'Domain {label}',
                  alpha=0.7, s=30, edgecolors='white', linewidths=0.5)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Feature Embedding ({method}, {var_explained:.1f}% var explained)')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_peak_family_plot(d_spacings: List[float], orientations: List[float], 
                          domain_id: int, path: str) -> None:
    """
    Save polar plot of peak families for a domain.
    
    Args:
        d_spacings: List of d-spacing values in nm
        orientations: List of orientation angles in degrees
        domain_id: Domain ID for title
        path: Output path for PNG
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if not d_spacings or not orientations:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.set_title(f'Domain {domain_id} - No peaks detected')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved (empty): {path}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Convert orientations to radians
    theta = np.deg2rad(orientations)
    
    # Use d-spacing as radial coordinate (inverted: larger d = closer to center)
    r = d_spacings
    
    # Plot points
    colors = plt.cm.viridis(np.linspace(0, 1, len(d_spacings)))
    scatter = ax.scatter(theta, r, c=colors, s=100, edgecolors='black', linewidths=1)
    
    # Add lines from origin to each point
    for t, radius, c in zip(theta, r, colors):
        ax.plot([0, t], [0, radius], color=c, linewidth=2, alpha=0.6)
    
    ax.set_title(f'Domain {domain_id} Peak Families\n({len(d_spacings)} peaks)', 
                 fontsize=12, pad=20)
    ax.set_theta_zero_location('E')  # 0° at right
    ax.set_theta_direction(1)  # Counter-clockwise
    
    # Label radial axis
    ax.set_ylabel('d-spacing (nm)', labelpad=30)
    
    # Add annotations for each peak
    for i, (t, radius, d) in enumerate(zip(theta, r, d_spacings)):
        ax.annotate(f'{d:.3f}nm', (t, radius), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_confidence_map(confidence: np.ndarray, path: str, 
                        title: str = "Clustering Confidence Map") -> None:
    """
    Save clustering confidence map.
    
    Args:
        confidence: 2D array of confidence values [0, 1]
        path: Output path for PNG
        title: Figure title
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1, origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
    
    ax.set_title(f"{title}\nMean: {np.mean(confidence):.3f}, "
                 f"Min: {np.min(confidence):.3f}")
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def save_gate_summary_plot(gates: dict, path: str) -> None:
    """
    Save visual summary of gate pass/fail status.
    
    Args:
        gates: Dictionary with gate results
        path: Output path for PNG
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    gate_names = list(gates.keys())
    n_gates = len(gate_names)
    
    fig, ax = plt.subplots(figsize=(10, max(4, n_gates * 0.5)))
    
    colors = []
    for g in gate_names:
        status = gates[g].get('status', 'UNKNOWN')
        if status == 'PASS':
            colors.append('#2ecc71')  # Green
        elif status == 'FAIL':
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#95a5a6')  # Gray
    
    y_pos = np.arange(n_gates)
    ax.barh(y_pos, [1] * n_gates, color=colors, height=0.6, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gate_names)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    
    # Add status text
    for i, g in enumerate(gate_names):
        status = gates[g].get('status', 'UNKNOWN')
        ax.text(0.5, i, status, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    ax.set_title('Gate Status Summary', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Top to bottom
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def create_pipeline_figure(image: np.ndarray, preprocessed: np.ndarray,
                          fft_power: np.ndarray, labels: np.ndarray,
                          path: str) -> None:
    """
    Create a multi-panel figure showing pipeline stages.
    
    Args:
        image: Original image
        preprocessed: Preprocessed image
        fft_power: Example FFT power spectrum
        labels: Domain labels
        path: Output path for PNG
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Original
    im1 = axes[0, 0].imshow(image, cmap='gray', origin='upper')
    axes[0, 0].set_title('(A) Original STEM-HAADF Image')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Preprocessed
    im2 = axes[0, 1].imshow(preprocessed, cmap='gray', origin='upper')
    axes[0, 1].set_title('(B) Preprocessed Image')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # FFT
    im3 = axes[1, 0].imshow(np.log1p(fft_power), cmap='inferno', origin='upper')
    axes[1, 0].set_title('(C) FFT Power Spectrum (log)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Domain labels
    im4 = axes[1, 1].imshow(labels, cmap='tab20', origin='upper')
    axes[1, 1].set_title(f'(D) Domain Labels (n={len(np.unique(labels))})')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    for ax in axes.flat:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    
    plt.suptitle('STEM-HAADF Crystal Domain Segmentation Pipeline', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")
