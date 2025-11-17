import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

class IRPTranslator:
    """
    Inverse Relative Projection (IRP) Translator
    Based on: "Latent Space Translation via Inverse Relative Projection"
    """
    
    def __init__(
        self, 
        scaler_X: StandardScaler, 
        scaler_Y: StandardScaler,
        omega: int = 8,           # Number of subspaces (ω in paper)
        delta: float = 0.7,       # Pruning threshold (δ in paper)
        ridge: float = 1e-4,      # Increased default ridge for stability
        use_anchor_completion: bool = False,
        verbose: bool = True
    ):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.omega = omega
        self.delta = delta
        self.ridge = ridge
        self.use_anchor_completion = use_anchor_completion
        self.verbose = verbose
        
    def _normalize_l2(self, X: np.ndarray) -> np.ndarray:
        """L2 normalize each row to unit norm (Eq. 1 in paper)"""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return X / norms
    
    def _customized_cosine_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Customized cosine distance from Eq. 3:
        d_cos(X) = 1 - |X · X^T / ||X||_2|
        """
        X_norm = self._normalize_l2(X)
        # Compute cosine similarity
        cos_sim = X_norm @ X_norm.T
        # Take absolute value to avoid quasi-colinear points with opposite directions
        cos_sim_abs = np.abs(cos_sim)
        # Return distance
        return 1.0 - cos_sim_abs
    
    def _farthest_point_sampling(
        self, 
        anchors: np.ndarray, 
        delta: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Greedy Farthest Point Sampling (FPS) with stopping condition.
        Returns indices of selected anchors.
        """
        if seed is not None:
            np.random.seed(seed)
            
        n = len(anchors)
        # Compute pairwise distances
        dist_matrix = self._customized_cosine_distance(anchors)
        
        # Start with random point
        selected_indices = [np.random.randint(n)]
        
        # Iteratively select farthest points
        for _ in range(n - 1):
            # Compute minimum distance from selected points
            min_dists = np.min(dist_matrix[selected_indices], axis=0)
            
            # Find farthest point
            farthest_idx = np.argmax(min_dists)
            max_dist = min_dists[farthest_idx]
            
            # Stopping condition: if maximum distance < delta, stop
            if max_dist < delta:
                break
                
            selected_indices.append(farthest_idx)
        
        return np.array(selected_indices)
    
    def _compute_relative_to_absolute_transform(self, A: np.ndarray) -> np.ndarray:
        """
        Compute transformation from relative to absolute space.
        
        Given anchors A (k, d), we want to go from relative coordinates (k,)
        to absolute coordinates (d,).
        
        If X_rel = X_abs @ A^T, then X_abs ≈ X_rel @ (A^T)^{-1}
        
        We compute (A^T)^{-1} using regularized pseudo-inverse.
        A^T has shape (d, k), so (A^T)^{-1} has shape (k, d).
        
        Returns: transform matrix of shape (k, d)
        """
        k, d = A.shape
        
        # Use SVD-based pseudo-inverse for numerical stability
        U, s, Vt = np.linalg.svd(A.T, full_matrices=False)  # A^T is (d, k)
        
        # Regularize singular values
        s_reg = s / (s**2 + self.ridge)
        
        # Compute pseudo-inverse: (A^T)^{-1} = V @ diag(s_reg) @ U^T
        A_T_inv = Vt.T @ np.diag(s_reg) @ U.T  # (k, d)
        
        return A_T_inv
    
    def fit(self, A_X: np.ndarray, A_Y: np.ndarray):
        """
        Fit the IRP translator using anchor points.
        
        Args:
            A_X: Anchor points in source space (k, d_X)
            A_Y: Corresponding anchor points in target space (k, d_Y)
        """
        if self.verbose:
            print(f"Fitting IRP with {len(A_X)} anchors...")
            print(f"  Omega (subspaces): {self.omega}")
            print(f"  Delta (pruning threshold): {self.delta}")
            print(f"  Ridge regularization: {self.ridge}")
        
        # Scale anchors
        A_X_scaled = self.scaler_X.transform(A_X)
        A_Y_scaled = self.scaler_Y.transform(A_Y)
        
        k, d_X = A_X_scaled.shape
        _, d_Y = A_Y_scaled.shape
        
        # Warn if anchor completion is enabled with different dimensions
        if self.use_anchor_completion and d_X != d_Y:
            print(f"  WARNING: Anchor completion disabled (d_X={d_X} != d_Y={d_Y})")
            self.use_anchor_completion = False
        
        # Center anchors (important for relative representations)
        self.mean_AX = A_X_scaled.mean(axis=0, keepdims=True)
        self.mean_AY = A_Y_scaled.mean(axis=0, keepdims=True)
        
        A_Xc = A_X_scaled - self.mean_AX
        A_Yc = A_Y_scaled - self.mean_AY
        
        # L2 normalize (Eq. 1: samples are rescaled to unit norm)
        A_Xc_norm = self._normalize_l2(A_Xc)
        A_Yc_norm = self._normalize_l2(A_Yc)
        
        # Store scale statistics for later
        self.scale_X = np.linalg.norm(A_Xc, axis=1).mean()
        self.scale_Y = np.linalg.norm(A_Yc, axis=1).mean()
        
        if self.verbose:
            print(f"  Mean scale X: {self.scale_X:.4f}")
            print(f"  Mean scale Y: {self.scale_Y:.4f}")
        
        # Create multiple anchor subspaces (Eq. 5: ensemble)
        self.subspaces = []
        condition_numbers_X = []
        condition_numbers_Y = []
        
        for i in range(self.omega):
            # Anchor pruning with different random seeds
            selected_indices = self._farthest_point_sampling(
                A_Xc_norm, 
                self.delta,
                seed=i
            )
            
            if len(selected_indices) < 2:
                if self.verbose:
                    print(f"  WARNING: Subspace {i} has too few anchors ({len(selected_indices)}), skipping")
                continue
            
            if self.verbose and i == 0:
                print(f"  Anchor pruning: {len(selected_indices)}/{k} anchors selected per subspace")
            
            S_X = A_Xc_norm[selected_indices]  # (k', d_X)
            S_Y = A_Yc_norm[selected_indices]  # (k', d_Y)
            
            k_prime = len(selected_indices)
            
            try:
                # Compute transformations
                # For encoding: X_abs (N, d_X) @ S_X^T (d_X, k') = X_rel (N, k')
                # For decoding: X_rel (N, k') @ R_Y (k', d_Y) = Y_abs (N, d_Y)
                
                # R_Y = (S_Y^T)^{-1} computed using SVD for stability
                R_Y = self._compute_relative_to_absolute_transform(S_Y)  # (k', d_Y)
                
                # For completeness, also compute R_X (though we use S_X^T directly for encoding)
                # S_X^T has shape (d_X, k')
                
                # Compute condition numbers for diagnostics
                U_X, s_X, _ = np.linalg.svd(S_X, full_matrices=False)
                U_Y, s_Y, _ = np.linalg.svd(S_Y, full_matrices=False)
                
                cond_X = s_X[0] / (s_X[-1] + 1e-10)
                cond_Y = s_Y[0] / (s_Y[-1] + 1e-10)
                
                condition_numbers_X.append(cond_X)
                condition_numbers_Y.append(cond_Y)
                
                self.subspaces.append({
                    'S_X': S_X,      # (k', d_X) - for encoding via @ S_X^T
                    'R_Y': R_Y,      # (k', d_Y) - for decoding
                    'cond_X': cond_X,
                    'cond_Y': cond_Y,
                    'n_anchors': k_prime
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Subspace {i} failed: {e}")
                continue
        
        if len(self.subspaces) == 0:
            raise RuntimeError("No valid subspaces created. Try reducing delta or increasing ridge.")
        
        # Compute average condition numbers
        avg_cond_X = np.mean(condition_numbers_X)
        avg_cond_Y = np.mean(condition_numbers_Y)
        
        if self.verbose:
            print(f"  Created {len(self.subspaces)} valid subspaces")
            print(f"  Average condition number X: {avg_cond_X:.2e}")
            print(f"  Average condition number Y: {avg_cond_Y:.2e}")
        
        # Validation: reconstruct anchors
        try:
            A_Yc_hat = self._translate_normalized(A_Xc_norm)
            rec_mse = np.mean((A_Yc_hat - A_Yc_norm) ** 2)
            
            # Compute cosine similarity
            cos_sim_matrix = cosine_similarity(A_Yc_hat, A_Yc_norm)
            rec_cos_sim = np.mean(np.diag(cos_sim_matrix))
            
            self.fit_rec_mse_ = rec_mse
            self.fit_rec_cos_sim_ = rec_cos_sim
            
            if self.verbose:
                print(f"  Anchor reconstruction MSE: {rec_mse:.6f}")
                print(f"  Anchor reconstruction Cosine Sim: {rec_cos_sim:.4f}")
        except Exception as e:
            if self.verbose:
                print(f"  WARNING: Could not compute reconstruction metrics: {e}")
            self.fit_rec_mse_ = None
            self.fit_rec_cos_sim_ = None
    
    def _translate_normalized(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Translate normalized data using ensemble of subspaces (Eq. 5).
        
        Translation pipeline:
        1. X_norm (N, d_X) @ S_X^T (d_X, k') -> X_rel (N, k')
        2. X_rel (N, k') @ R_Y (k', d_Y) -> Y_norm (N, d_Y)
        """
        predictions = []
        
        for subspace in self.subspaces:
            # Encode to relative space: project onto anchors
            X_rel = X_norm @ subspace['S_X'].T  # (N, d_X) @ (d_X, k') = (N, k')
            
            # Decode to target space: map from relative to absolute
            Y_norm = X_rel @ subspace['R_Y']  # (N, k') @ (k', d_Y) = (N, d_Y)
            
            predictions.append(Y_norm)
        
        # Average pool (Eq. 5: parameter-free ensemble)
        Y_norm_pred = np.mean(predictions, axis=0)
        
        return Y_norm_pred
    
    def translate(self, X: np.ndarray) -> np.ndarray:
        """
        Translate samples from source space to target space.
        
        Args:
            X: Input samples in source space (N, d_X)
            
        Returns:
            Y: Translated samples in target space (N, d_Y)
        """
        if not hasattr(self, 'subspaces'):
            raise RuntimeError("Call fit() first")
        
        # Scale and center
        X_scaled = self.scaler_X.transform(X)
        Xc = X_scaled - self.mean_AX
        
        # L2 normalize
        X_norm = self._normalize_l2(Xc)
        
        # Translate using ensemble
        Y_norm_pred = self._translate_normalized(X_norm)
        
        # Rescale to target space scale (Section 3.4: scale invariance)
        # The paper shows that the scale can be ignored, but we restore it here
        Y_pred_scaled = Y_norm_pred * self.scale_Y
        
        # De-center and de-scale
        Y_pred_scaled = Y_pred_scaled + self.mean_AY
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        
        return Y_pred