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
        omega: int = 8,           
        delta: float = 0.7,       
        ridge: float = 1e-4,      
        verbose: bool = True
    ):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.omega = omega
        self.delta = delta
        self.ridge = ridge
        self.verbose = verbose
        
    def _normalize_l2(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms
    
    def _customized_cosine_distance(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize_l2(X)
        cos_sim = X_norm @ X_norm.T
        cos_sim_abs = np.abs(cos_sim)
        return 1.0 - cos_sim_abs
    
    def _farthest_point_sampling(
        self, 
        anchors: np.ndarray, 
        delta: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        n = len(anchors)
        dist_matrix = self._customized_cosine_distance(anchors)
        selected_indices = [np.random.randint(n)]
        
        for _ in range(n - 1):
            min_dists = np.min(dist_matrix[selected_indices], axis=0)
            farthest_idx = np.argmax(min_dists)
            max_dist = min_dists[farthest_idx]
            
            if max_dist < delta:
                break
            selected_indices.append(farthest_idx)
        
        return np.array(selected_indices)
    
    def _compute_relative_to_absolute_transform(self, A: np.ndarray) -> np.ndarray:
        k, d = A.shape
        U, s, Vt = np.linalg.svd(A.T, full_matrices=False)
        s_reg = s / (s**2 + self.ridge)
        A_T_inv = Vt.T @ np.diag(s_reg) @ U.T
        return A_T_inv
    
    def fit(self, A_X: np.ndarray, A_Y: np.ndarray):
        if self.verbose:
            print(f"Fitting IRP with {len(A_X)} anchors...")
            print(f"  Omega (subspaces): {self.omega}")
            print(f"  Delta (pruning threshold): {self.delta}")
            print(f"  Ridge regularization: {self.ridge}")
        
        A_X_scaled = self.scaler_X.transform(A_X)
        A_Y_scaled = self.scaler_Y.transform(A_Y)
        
        k, d_X = A_X_scaled.shape
        _, d_Y = A_Y_scaled.shape
        
        self.mean_AX = A_X_scaled.mean(axis=0, keepdims=True)
        self.mean_AY = A_Y_scaled.mean(axis=0, keepdims=True)
        
        A_Xc = A_X_scaled - self.mean_AX
        A_Yc = A_Y_scaled - self.mean_AY
        
        A_Xc_norm = self._normalize_l2(A_Xc)
        A_Yc_norm = self._normalize_l2(A_Yc)
        
        self.scale_X = np.linalg.norm(A_Xc, axis=1).mean()
        self.scale_Y = np.linalg.norm(A_Yc, axis=1).mean()
        
        if self.verbose:
            print(f"  Mean scale X: {self.scale_X:.4f}")
            print(f"  Mean scale Y: {self.scale_Y:.4f}")
        
        self.subspaces = []
        
        for i in range(self.omega):
            selected_indices = self._farthest_point_sampling(
                A_Xc_norm, self.delta, seed=i
            )
            
            if len(selected_indices) < 2:
                continue
            
            if self.verbose and i == 0:
                print(f"  Anchor pruning: {len(selected_indices)}/{k} anchors selected per subspace")
            
            S_X = A_Xc_norm[selected_indices]
            S_Y = A_Yc_norm[selected_indices]
            
            try:
                R_Y = self._compute_relative_to_absolute_transform(S_Y)
                self.subspaces.append({'S_X': S_X, 'R_Y': R_Y})
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Subspace {i} failed: {e}")
                continue
        
        if len(self.subspaces) == 0:
            raise RuntimeError("No valid subspaces created. Try reducing delta or increasing ridge.")
        
        if self.verbose:
            print(f"  Created {len(self.subspaces)} valid subspaces")

    def _translate_normalized(self, X_norm: np.ndarray) -> np.ndarray:
        predictions = []
        for subspace in self.subspaces:
            X_rel = X_norm @ subspace['S_X'].T
            Y_norm = X_rel @ subspace['R_Y']
            predictions.append(Y_norm)
        Y_norm_pred = np.mean(predictions, axis=0)
        return Y_norm_pred
    
    def translate(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'subspaces'):
            raise RuntimeError("Call fit() first")
        
        X_scaled = self.scaler_X.transform(X)
        Xc = X_scaled - self.mean_AX
        X_norm = self._normalize_l2(Xc)
        
        Y_norm_pred = self._translate_normalized(X_norm)
        
        Y_pred_scaled = Y_norm_pred * self.scale_Y
        Y_pred_scaled = Y_pred_scaled + self.mean_AY
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        
        return Y_pred