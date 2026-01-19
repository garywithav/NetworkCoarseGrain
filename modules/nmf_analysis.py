import numpy as np
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt 
import seaborn as sns


# Example function to apply NMF to an estimated Aij matrix

def apply_nmf_to_Aij(Aij, n_components=2, random_state=123):
    """
    Apply Non-negative Matrix Factorization (NMF) to the given Aij matrix.
    Args:
        Aij (np.ndarray): The estimated Aij matrix (non-negative).
        n_components (int): Number of latent components to extract.
        random_state (int): Random seed for reproducibility.
    Returns:
        W (np.ndarray): Basis matrix.
        H (np.ndarray): Coefficient matrix.
        model (NMF): The fitted NMF model.
    """
    model = NMF(n_components=n_components, init='nndsvda', random_state=random_state)
    W = model.fit_transform(Aij)
    H = model.components_
    return W, H, model


def analyze_nmf_results(W, H, save_prefix='nmf_analysis'):
    """
    General NMF analysis for any sample size, deme size, and number of components.
    
    Parameters:
    -----------
    W : np.ndarray, shape (n_samples, n_components)
        NMF weight matrix
    H : np.ndarray, shape (n_components, n_features)
        NMF component matrix
    save_prefix : str
        Prefix for saved figure filenames
    """
    
    n_samples, n_components = W.shape
    n_features = H.shape[1]
    
    # Infer deme size (assume square matrix was flattened)
    n_demes = int(np.sqrt(n_features))
    
    if n_demes ** 2 != n_features:
        raise ValueError(f"n_features={n_features} is not a perfect square. Cannot infer deme structure.")
    
    print("=" * 80)
    print(f"NMF ANALYSIS: {n_samples} samples, {n_demes}×{n_demes} demes, {n_components} components")
    print("=" * 80)
    
    # ========================================================================
    # 1. VISUALIZE ALL DISCOVERED PATTERNS
    # ========================================================================
    n_cols = min(4, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_components == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_components):
        pattern = H[i, :].reshape(n_demes, n_demes)
        
        # Only annotate if small enough to be readable
        annotate = (n_demes <= 15)
        
        sns.heatmap(pattern, ax=axes[i], cmap='viridis', 
                    annot=annotate, fmt='.2f' if annotate else '',
                    cbar=True, square=True,
                    annot_kws={'size': 8} if annotate else {},
                    cbar_kws={'shrink': 0.8},
                    xticklabels=False, yticklabels=False)
        axes[i].set_title(f'Component {i+1}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('To Deme', fontsize=10)
        axes[i].set_ylabel('From Deme', fontsize=10)
    
    # Hide extra subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{n_components} Discovered Migration Patterns ({n_demes}×{n_demes} demes)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_patterns.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 2. COMPONENT PREVALENCE
    # ========================================================================
    component_prevalence = W.mean(axis=0)
    component_std = W.std(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    ax1.bar(range(1, n_components + 1), component_prevalence, 
            yerr=component_std, capsize=5, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Component', fontsize=12)
    ax1.set_ylabel('Average Weight (± std)', fontsize=12)
    ax1.set_title(f'Component Prevalence Across {n_samples} Samples', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, n_components + 1))
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot([W[:, i] for i in range(n_components)], 
                      labels=range(1, n_components + 1),
                      patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xlabel('Component', fontsize=12)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.set_title('Distribution of Component Weights', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_prevalence.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 3. STATISTICS TABLE
    # ========================================================================
    print("\nCOMPONENT STATISTICS:")
    print("-" * 80)
    print(f"{'Comp':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} "
          f"{'%NonZero':<10} {'Pattern Sparsity':<15}")
    print("-" * 80)
    
    for i in range(n_components):
        pattern = H[i, :]
        weights = W[:, i]
        sparsity = (pattern < 0.001).sum() / len(pattern) * 100
        
        print(f"{i+1:<6} {weights.mean():<10.4f} {weights.std():<10.4f} "
              f"{weights.min():<10.4f} {weights.max():<10.4f} "
              f"{(weights > 0.001).sum()/len(weights)*100:<10.1f} "
              f"{sparsity:<15.1f}%")
    
    # ========================================================================
    # 4. COMPONENT CORRELATION
    # ========================================================================
    if n_components > 1:
        correlation = np.corrcoef(W.T)
        
        fig_size = max(6, min(12, n_components * 1.5))
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, cbar_kws={'label': 'Correlation'},
                    xticklabels=range(1, n_components+1),
                    yticklabels=range(1, n_components+1),
                    vmin=-1, vmax=1)
        plt.title('Component Co-occurrence\n(Do components appear together?)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Component', fontsize=12)
        plt.ylabel('Component', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_correlation.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    # ========================================================================
    # 5. SAMPLE DIVERSITY VISUALIZATION (if enough samples)
    # ========================================================================
    if n_samples >= 50:
        pca = PCA(n_components=min(2, n_components))
        W_pca = pca.fit_transform(W)
        
        plt.figure(figsize=(10, 8))
        
        if W_pca.shape[1] == 2:
            plt.scatter(W_pca[:, 0], W_pca[:, 1], alpha=0.3, s=10, c='steelblue')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        else:
            plt.hist(W_pca[:, 0], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
        
        plt.title(f'Sample Diversity in Component Space\n({n_samples} samples)', 
                  fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_sample_space.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    # ========================================================================
    # 6. EXAMPLE SAMPLE COMPOSITIONS (show a few representative samples)
    # ========================================================================
    print("\n\nEXAMPLE SAMPLE COMPOSITIONS:")
    print("-" * 80)
    
    n_examples = min(5, n_samples)
    example_indices = np.linspace(0, n_samples-1, n_examples, dtype=int)
    
    for idx in example_indices:
        print(f"\nSample {idx}:")
        for comp in range(n_components):
            bar_length = int(W[idx, comp] * 50)
            print(f"  Component {comp+1}: {W[idx, comp]:6.4f} {'█' * bar_length}")
    
    # ========================================================================
    # 7. RECONSTRUCTION ERROR
    # ========================================================================
    reconstruction = W @ H
    mse = np.mean((reconstruction - reconstruction) ** 2)  # You'd compare to original data
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Figures saved with prefix: {save_prefix}")
    print("=" * 80)
    
    return {
        'n_samples': n_samples,
        'n_demes': n_demes,
        'n_components': n_components,
        'prevalence': component_prevalence,
        'std': component_std,
        'correlation': correlation if n_components > 1 else None
    }


if __name__ == "__main__":
    # Example usage: load an estimated Aij matrix and apply NMF
    # Replace 'path_to_Aij.npy' with your actual file path
    Aij = np.load('path_to_Aij.npy')
    W, H, nmf_model = apply_nmf_to_Aij(Aij, n_components=2)
    print("W (basis):", W)
    print("H (coefficients):", H)
