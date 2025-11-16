import numpy as np
import torch

# --- Base Metrics ---

def mrr(pred_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    reciprocal_ranks = []
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i] == gt_indices[i])[0]
        if matches.size > 0:
            reciprocal_ranks.append(1.0 / (matches[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

def recall_at_k(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int) -> float:
    recall = 0
    for i in range(len(gt_indices)):
        if gt_indices[i] in pred_indices[i, :k]:
            recall += 1
    recall /= len(gt_indices)
    return recall

def ndcg(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int = 100) -> float:
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