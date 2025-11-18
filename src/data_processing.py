import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .baseline_utils import load_data, prepare_train_data
from . import config
import gc

def find_all_noisy_captions(data_raw, similarity_threshold=0.50):
    """
    Identifies noisy captions based on cosine similarity to the centroid of
    their respective image embeddings.
    """

    print(f"Threshold = {similarity_threshold}")
    caption_embd = data_raw['captions/embeddings']
    image_embd = data_raw['images/embeddings']
    label = data_raw['captions/label']

    num_images = image_embd.shape[0]
    noisy_indices_set = set() 

    for j in tqdm(range(num_images), desc="Analysing images: "):
        caption_indices = np.where(label[:, j])[0]

        if len(caption_indices) < 2:
            continue

        image_caption_embeddings = caption_embd[caption_indices]
        centroid = np.mean(image_caption_embeddings, axis=0, keepdims=True)
        similarities = cosine_similarity(image_caption_embeddings, centroid).flatten()

        for i, sim_score in enumerate(similarities):
            if sim_score < similarity_threshold:
                global_caption_index = caption_indices[i]
                noisy_indices_set.add(global_caption_index)

    print(f"\\n--- Noise Analysis Completed ---")
    print(f"Found {len(noisy_indices_set)} noisy samples (with threshold < {similarity_threshold}).")
    return noisy_indices_set

def load_and_clean_data(data_path, noise_threshold):
    """
    Loads, prepares, and cleans the training data by removing noise.
    """
    # 1. Load Data
    print("1. Load Data...")
    train_data_raw = load_data(data_path)
    X_totali_torch, Y_totali_torch, _ = prepare_train_data(train_data_raw)
    print(f"Total samples loaded: {X_totali_torch.shape[0]}.")

    # 2. Noise Analysis and Filtering
    noisy_indices = find_all_noisy_captions(train_data_raw, similarity_threshold=noise_threshold)

    # 3. Create Clean Data
    print(f"Removing {len(noisy_indices)} noisy samples...")
    mask = np.ones(X_totali_torch.shape[0], dtype=bool)
    mask[list(noisy_indices)] = False

    X_clean_np = X_totali_torch.numpy()[mask]
    Y_clean_np = Y_totali_torch.numpy()[mask]

    print(f"Clean data ready: {X_clean_np.shape[0]} samples.")

    # Clean up memory
    del train_data_raw, X_totali_torch, Y_totali_torch

    return X_clean_np, Y_clean_np


def load_and_prep_data_direct(train_path, coco_path, use_coco, noise_threshold, val_split_ratio, random_seed):
    """
    Implements the full data loading, merging, cleaning, and splitting
    logic from the second notebook (aml-notebook_finale.ipynb, Cell 4).
    
    This function prepares data for full-retrieval evaluation.
    """
    print("\n--- 1. Set seed ---")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("\n--- 2. Load Dataset ---")
    if use_coco:
        # (Insert the entire 'if USE_COCO_DATASET:' block from Cell 4 here)
        # ...
        print("Using coco: merging the two datasets...")
        # ... (concatenation logic) ...
        pass # Placeholder for the large block
    else:
        print(f"Using just Original Dataset")
        print(f"Loading dataset: {train_path}")
        train_data = load_data(train_path)
        captions_text = train_data['captions/text']
        caption_embd_np = train_data['captions/embeddings']
        gallery_images_names = train_data['images/names']
        gallery_images_embeddings = train_data['images/embeddings']
        label_mat = train_data['captions/label']
        print(f"  Dataset loaded: {label_mat.shape[0]} captions, {label_mat.shape[1]} images.")

    print("\n--- 3. Data Preparation and Noise Cleaning ---")
    train_data_dict = {
        'captions/text': captions_text, 'captions/embeddings': caption_embd_np, 
        'images/embeddings': gallery_images_embeddings, 'captions/label': label_mat,
        'images/names': gallery_images_names
    }
    
    X, y, label_tensor = prepare_train_data(train_data_dict)
    print(f"Initial dataset size: {len(X)} pairs.")
    del train_data_dict
    if 'train_data' in locals(): del train_data
    del label_tensor
    gc.collect()

    mask = np.ones(len(X), dtype=bool) 
    if config.CLEAN_DATA:
        print(f"\\nStarting noise cleaning (Threshold < {noise_threshold})...")
        num_images = gallery_images_embeddings.shape[0]
        noisy_indices_set = set()
        for j in tqdm(range(num_images), desc="Noise Analysis"):
            cap_idx = np.where(label_mat[:, j])[0]
            if len(cap_idx) < 2: continue 
            group_embds = caption_embd_np[cap_idx]
            centroid = np.mean(group_embds, axis=0, keepdims=True)
            sims = cosine_similarity(group_embds, centroid).flatten()
            for i, s in enumerate(sims):
                if s < noise_threshold:
                    noisy_indices_set.add(cap_idx[i])
        print(f"Identified {len(noisy_indices_set)} noisy samples.")
        mask[list(noisy_indices_set)] = False
    del caption_embd_np, noisy_indices_set
    gc.collect()

    print("\n--- 4. Applying Filter and Train/Val Split ---")
    X = X[mask]
    y = y[mask]
    label = torch.from_numpy(label_mat[mask])
    captions_text_clean = captions_text[mask]
    del label_mat, mask
    gc.collect()

    DATASET_SIZE = len(X)
    print(f"Clean dataset: {DATASET_SIZE} pairs.")
    
    n_train = int((1 - val_split_ratio) * DATASET_SIZE)
    TRAIN_SPLIT = torch.zeros(DATASET_SIZE, dtype=torch.bool)
    TRAIN_SPLIT[:n_train] = 1

    X_train, X_val = X[TRAIN_SPLIT], X[~TRAIN_SPLIT]
    y_train, y_val = y[TRAIN_SPLIT], y[~TRAIN_SPLIT]
    print(f"Training Set:    X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Set: X={X_val.shape}, y={y_val.shape}")
    del X, y
    gc.collect()

    print("\n--- 5. Preparing Gallery for Retrieval ---")
    label_cpu = label.cpu() 
    img_VAL_SPLIT = label_cpu[~TRAIN_SPLIT].sum(dim=0) > 0
    del label_cpu
    
    val_text_embd = X_val
    val_img_embd_unique = torch.from_numpy(gallery_images_embeddings[img_VAL_SPLIT.numpy()])
    del gallery_images_embeddings
    gc.collect()

    val_submatrix = label[~TRAIN_SPLIT][:, img_VAL_SPLIT]
    val_label_gt = np.argmax(val_submatrix.cpu().numpy(), axis=1)
    del val_submatrix, label
    gc.collect()

    print(f"Query (Captions):      {len(val_text_embd)}")
    print(f"Gallery (Images):      {len(val_img_embd_unique)}")
    print(f"Ground Truth Indices: {val_label_gt.shape}")
    
    # Return all the necessary components
    return X_train, y_train, X_val, y_val, val_text_embd, val_img_embd_unique, val_label_gt