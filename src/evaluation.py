import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# --- Base Metrics ---

def mrr(pred_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for retrieval results.
    Args:
        pred_indices (np.ndarray): Predicted ranked indices for each query.
        gt_indices (np.ndarray): Ground truth indices for each query.
    Returns:
        float: Mean reciprocal rank score.
    """
    reciprocal_ranks = []
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i] == gt_indices[i])[0]
        if matches.size > 0:
            reciprocal_ranks.append(1.0 / (matches[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

def recall_at_k(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int) -> float:
    """
    Computes Recall@k for retrieval results.
    Args:
        pred_indices (np.ndarray): Predicted ranked indices for each query.
        gt_indices (np.ndarray): Ground truth indices for each query.
        k (int): Top-k to consider for recall.
    Returns:
        float: Recall@k score.
    """
    recall = 0
    for i in range(len(gt_indices)):
        if gt_indices[i] in pred_indices[i, :k]:
            recall += 1
    recall /= len(gt_indices)
    return recall

def ndcg(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int = 100) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at k for retrieval results.
    Args:
        pred_indices (np.ndarray): Predicted ranked indices for each query.
        gt_indices (np.ndarray): Ground truth indices for each query.
        k (int): Top-k to consider for NDCG.
    Returns:
        float: NDCG@k score.
    """
    ndcg_total = 0.0
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i, :k] == gt_indices[i])[0]
        if matches.size > 0:
            rank = matches[0] + 1
            ndcg_total += 1.0 / np.log2(rank + 1)
    return ndcg_total / len(gt_indices)

# --- Main Evaluation Function ---

@torch.inference_mode()
def evaluate_retrieval(translated_embd, image_embd, gt_indices, max_indices = 99, batch_size=100):
    """
    Evaluates retrieval performance using several metrics (MRR, NDCG, Recall@k, L2 distance).
    Args:
        translated_embd (np.ndarray or torch.Tensor): Query/caption embeddings.
        image_embd (np.ndarray or torch.Tensor): Gallery/image embeddings.
        gt_indices (np.ndarray): Ground truth indices for each query.
        max_indices (int): Number of top indices to consider for metrics.
        batch_size (int): Batch size for evaluation.
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()

    n_queries = translated_embd.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translated_embd = translated_embd.to(device)
    image_embd = image_embd.to(device)

    all_sorted_indices = []
    l2_distances = []

    for start_idx in range(0, n_queries, batch_size):
        batch_slice = slice(start_idx, min(start_idx + batch_size, n_queries))
        batch_translated = translated_embd[batch_slice]
        batch_img_embd = image_embd[batch_slice]

        batch_similarity = batch_translated @ batch_img_embd.T
        batch_indices = batch_similarity.topk(k=max_indices, dim=1, sorted=True).indices.cpu().numpy()

        global_indices_map = np.arange(start_idx, min(start_idx + batch_size, n_queries))
        mapped_indices = global_indices_map[batch_indices]
        all_sorted_indices.append(mapped_indices)

        batch_gt_indices_global = gt_indices[batch_slice]
        batch_gt_embeddings = image_embd[batch_gt_indices_global]
        batch_l2 = (batch_translated - batch_gt_embeddings).norm(dim=1).cpu()
        l2_distances.append(batch_l2)

    sorted_indices = np.concatenate(all_sorted_indices, axis=0)

    metrics = {
        'mrr': mrr,
        'ndcg': ndcg,
        'recall_at_1': lambda preds, gt: recall_at_k(preds, gt, 1),
        'recall_at_3': lambda preds, gt: recall_at_k(preds, gt, 3),
        'recall_at_5': lambda preds, gt: recall_at_k(preds, gt, 5),
        'recall_at_10': lambda preds, gt: recall_at_k(preds, gt, 10),
    }

    results = {
        name: func(sorted_indices, gt_indices)
        for name, func in metrics.items()
    }

    l2_dist = torch.cat(l2_distances, dim=0).mean().item()
    results['l2_dist'] = l2_dist

    return results

# --- Wrapper from Notebook 2 (for evaluation) ---
class MLPWrapper:
    """ Wrapper for evaluation. Always normalizes output for Cosine Similarity. """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval() 
        
    @torch.inference_mode()
    def translate(self, x_data, batch_size=1024):
        """
        Translates input data using the wrapped model and normalizes the output.
        Args:
            x_data (np.ndarray or torch.Tensor): Input data to translate.
            batch_size (int): Batch size for inference.
        Returns:
            np.ndarray: Normalized model outputs.
        """
        self.model.eval()
        outputs = []
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data).float()
            
        loader = DataLoader(TensorDataset(x_data), batch_size=batch_size, shuffle=False)
        for (bx,) in loader:
            batch_x = bx.float().to(self.device)
            pred_raw = self.model(batch_x)
            pred_norm = F.normalize(pred_raw, p=2, dim=1)
            outputs.append(pred_norm.cpu().numpy())
        return np.vstack(outputs)

# --- Full Retrieval Evaluation (N-vs-M) from Notebook 2 ---
@torch.inference_mode()
def evaluate_retrieval_full(
    translated_embd, image_embd, gt_indices,
    max_k=50, batch_size=256, device=None
):
    """
    Full-dataset retrieval evaluation (N queries vs M gallery items).
    Computes MRR, NDCG, Recall@k, and L2 distance for all queries.
    Args:
        translated_embd (np.ndarray or torch.Tensor): Query/caption embeddings.
        image_embd (np.ndarray or torch.Tensor): Gallery/image embeddings.
        gt_indices (np.ndarray): Ground truth indices for each query.
        max_k (int): Maximum k for top-k metrics.
        batch_size (int): Batch size for evaluation.
        device (torch.device, optional): Device for computation.
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    translated_embd = translated_embd.to(device)
    image_embd = image_embd.to(device)
    N_queries = translated_embd.shape[0]
    gt_indices = np.asarray(gt_indices)

    sorted_indices_global = []
    l2_accum = []

    for start in range(0, N_queries, batch_size):
        end = min(start + batch_size, N_queries)
        batch_t = F.normalize(translated_embd[start:end], p=2, dim=1)
        
        # NOTE: Assumes image_embd is the *full gallery* and is pre-normalized
        sims = batch_t @ image_embd.T  # (b, M_gallery)
        
        topk_idx = sims.topk(k=max_k, dim=1).indices.cpu().numpy()
        sorted_indices_global.append(topk_idx)
        
        gt_slice = gt_indices[start:end]
        gt_emb = image_embd[gt_slice]
        l2_batch = (batch_t - gt_emb).norm(dim=1).cpu()
        l2_accum.append(l2_batch)

    sorted_indices = np.vstack(sorted_indices_global)
    l2_mean = torch.cat(l2_accum).mean().item()
    
    # ... (Insert _mrr, _recall_at_k, _ndcg sub-functions from Cell 5) ...
    def _mrr(pred, gt):
        rr = []
        for i in range(len(gt)):
            match = np.where(pred[i] == gt[i])[0]
            rr.append(1.0 / (match[0] + 1) if len(match) > 0 else 0.0)
        return np.mean(rr)

    def _recall_at_k(pred, gt, k):
        return np.mean([gt[i] in pred[i, :k] for i in range(len(gt))])

    def _ndcg(pred, gt, k=100):
        scores = []
        for i in range(len(gt)):
            match = np.where(pred[i, :k] == gt[i])[0]
            if len(match) > 0:
                rank = match[0] + 1
                scores.append(1.0 / np.log2(rank + 1))
            else:
                scores.append(0.0)
        return float(np.mean(scores))

    metrics = {
        "mrr": _mrr(sorted_indices, gt_indices),
        "ndcg": _ndcg(sorted_indices, gt_indices, max_k),
        "l2_dist": l2_mean,
    }
    for k in [1, 3, 5, 10, 20, 50]:
        if k <= max_k:
            metrics[f"recall@{k}"] = _recall_at_k(sorted_indices, gt_indices, k)
    return metrics

# --- In-Batch Retrieval (N-vs-N) from Notebook 2 ---
@torch.inference_mode()
def aml_inbatch_retrieval(pred_embeds, target_embeds, batch_size=100):
    """
    Computes in-batch retrieval metrics (R@1, R@5, R@10, MRR) for N-vs-N retrieval.
    Args:
        pred_embeds (np.ndarray or torch.Tensor): Predicted embeddings.
        target_embeds (np.ndarray or torch.Tensor): Target embeddings.
        batch_size (int): Batch size for evaluation.
    Returns:
        dict: Dictionary with r1, r5, r10, and mrr scores.
    """

    pred = F.normalize(pred_embeds, p=2, dim=1)
    targ = F.normalize(target_embeds, p=2, dim=1)
    N = pred.shape[0]
    r1, r5, r10, mrr, n_batches = 0, 0, 0, 0, 0

    for i in range(0, N, batch_size):
        end = i + batch_size
        if end > N: break
        p = pred[i:end]
        t = targ[i:end]
        sims = p @ t.T
        ranks = (sims > sims.diag().unsqueeze(1)).sum(dim=1) + 1
        r1  += (ranks == 1).sum().item()
        r5  += (ranks <= 5).sum().item()
        r10 += (ranks <= 10).sum().item()
        mrr += (1.0 / ranks.float()).sum().item()
        n_batches += 1

    total_samples = n_batches * batch_size
    if total_samples == 0: return {"r1": 0, "r5": 0, "r10": 0, "mrr": 0}
    return {
        "r1":  r1  / total_samples, "r5":  r5  / total_samples,
        "r10": r10 / total_samples, "mrr": mrr / total_samples
    }
    