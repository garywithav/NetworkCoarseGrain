import numpy as np
from sklearn.decomposition import NMF

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

if __name__ == "__main__":
    # Example usage: load an estimated Aij matrix and apply NMF
    # Replace 'path_to_Aij.npy' with your actual file path
    Aij = np.load('path_to_Aij.npy')
    W, H, nmf_model = apply_nmf_to_Aij(Aij, n_components=2)
    print("W (basis):", W)
    print("H (coefficients):", H)
