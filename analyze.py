"""
Analyze trained bilinear layer using tensor decomposition.

Computes the 3rd-order tensor representation via weight contraction,
then performs SVD analysis following Section 3.3 of the bilinear paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train import BilinearLayer

def compute_interaction_tensor(model):
    """
    Contract bilinear weights into 3rd-order tensor.
    T[i,j,k] = sum_h (W1[h,i] * W2[h,j] * W_out[k,h])
    """
    W1 = model.W1.weight.data.numpy()
    W2 = model.W2.weight.data.numpy()
    W_out = model.W_out.weight.data.numpy()
    
    print(f"Computing 3rd-order tensor via einsum contraction...")
    tensor_3d = np.einsum('hi,hj,kh->ijk', W1, W2, W_out)
    print(f"Tensor shape: {tensor_3d.shape}")
    
    return tensor_3d


def compute_svd_decomposition(tensor_3d, n_components=5):
    """Perform SVD on mode-1 and mode-2 unfoldings."""
    d1, d2, d3 = tensor_3d.shape
    
    # Mode unfoldings
    mode1 = tensor_3d.reshape(d1, -1)
    mode2 = tensor_3d.transpose(1, 0, 2).reshape(d2, -1)
    
    print(f"Computing SVD decomposition...")
    U1, S1, _ = np.linalg.svd(mode1, full_matrices=False)
    U2, S2, _ = np.linalg.svd(mode2, full_matrices=False)
    
    return {
        'mode1_vectors': U1[:, :n_components],
        'mode1_values': S1[:n_components],
        'mode2_vectors': U2[:, :n_components],
        'mode2_values': S2[:n_components]
    }


def plot_interaction_matrices(tensor_3d):
    """Visualize slices of the 3rd-order tensor."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    output_indices = np.linspace(0, tensor_3d.shape[2]-1, 9, dtype=int)
    
    for idx, ax in enumerate(axes.flat):
        k = output_indices[idx]
        matrix = tensor_3d[:, :, k]
        vmax = np.abs(matrix).max()
        
        im = ax.imshow(matrix, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Output {k}', fontsize=10)
        ax.set_xlabel('Input j')
        ax.set_ylabel('Input i')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Interaction Matrices T[i,j,k]', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/interaction_matrices.png', dpi=150, bbox_inches='tight')
    print("Saved: results/interaction_matrices.png")


def plot_eigenvector_analysis(svd_data):
    """Visualize eigenvector components and singular values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mode-1 components
    ax = axes[0, 0]
    for i in range(5):
        ax.plot(svd_data['mode1_vectors'][:, i], label=f'Comp {i+1}', alpha=0.7)
    ax.set_title('Mode-1 Eigenvector Components')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mode-2 components
    ax = axes[0, 1]
    for i in range(5):
        ax.plot(svd_data['mode2_vectors'][:, i], label=f'Comp {i+1}', alpha=0.7)
    ax.set_title('Mode-2 Eigenvector Components')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Singular values
    ax = axes[1, 0]
    ax.bar(range(5), svd_data['mode1_values'])
    ax.set_title('Mode-1 Singular Values')
    ax.set_xlabel('Component')
    ax.set_ylabel('Value')
    ax.grid(alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.bar(range(5), svd_data['mode2_values'])
    ax.set_title('Mode-2 Singular Values')
    ax.set_xlabel('Component')
    ax.set_ylabel('Value')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/eigenvector_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: results/eigenvector_analysis.png")


def plot_eigenvector_heatmaps(svd_data):
    """Heatmap visualization of eigenvector structure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mode-1
    ax = axes[0]
    vmax = np.abs(svd_data['mode1_vectors']).max()
    im = ax.imshow(svd_data['mode1_vectors'].T, aspect='auto', cmap='RdBu',
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Component')
    ax.set_title('Mode-1 Eigenvectors')
    plt.colorbar(im, ax=ax)
    
    # Mode-2
    ax = axes[1]
    vmax = np.abs(svd_data['mode2_vectors']).max()
    im = ax.imshow(svd_data['mode2_vectors'].T, aspect='auto', cmap='RdBu',
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Component')
    ax.set_title('Mode-2 Eigenvectors')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('results/eigenvector_heatmaps.png', dpi=150, bbox_inches='tight')
    print("Saved: results/eigenvector_heatmaps.png")


def main():
    print("Analyzing Bilinear Layer via Tensor Decomposition")
    print("-" * 60)
    
    # Load model
    checkpoint = torch.load('bilinear_model.pt')
    P = checkpoint['P']
    model = BilinearLayer(checkpoint['input_dim'], checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model (P={P}, accuracy={checkpoint['final_accuracy']:.4f})")
    print("-" * 60)
    
    # Compute tensor representation
    tensor_3d = compute_interaction_tensor(model)
    
    # SVD analysis
    svd_data = compute_svd_decomposition(tensor_3d, n_components=5)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_interaction_matrices(tensor_3d)
    plot_eigenvector_analysis(svd_data)
    plot_eigenvector_heatmaps(svd_data)
    
    # Print summary
    print("\n" + "-" * 60)
    print("Analysis Summary:")
    print(f"  Mode-1 top singular value: {svd_data['mode1_values'][0]:.2f}")
    print(f"  Mode-2 top singular value: {svd_data['mode2_values'][0]:.2f}")
    ratio1 = svd_data['mode1_values'][0] / svd_data['mode1_values'][1]
    ratio2 = svd_data['mode2_values'][0] / svd_data['mode2_values'][1]
    print(f"  Ratio 1st/2nd (mode-1): {ratio1:.2f}")
    print(f"  Ratio 1st/2nd (mode-2): {ratio2:.2f}")
    print("\nLow ratio indicates distributed representation (not low-rank)")
    print("-" * 60)


if __name__ == "__main__":
    main()