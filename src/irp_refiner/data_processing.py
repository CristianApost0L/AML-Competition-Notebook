import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .baseline_utils import load_data, prepare_train_data

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