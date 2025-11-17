import torch

# --- General Config ---
SEED = 42
CHECKPOINT_DIR = "/kaggle/working/checkpoints/"
DATA_DIR = "/kaggle/input/aml-competition/"
TRAIN_DATA_PATH = f"{DATA_DIR}train/train/train.npz"
TEST_DATA_PATH = f"{DATA_DIR}test/test/test.clean.npz"

# --- Training Hyperparameters ---
EPOCHS = 200
BATCH_SIZE = 256
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SIZE = 0.20
EARLY_STOP_PATIENCE = 10
MIN_IMPROVEMENT_DELTA = 0.0001
K_FOLDS = 5

# --- Model Dimensions ---
D_X = 1536    # Text Embedding dim
D_Y = 1536    # Image Embedding dim

# --- MLP Architecture ---
NUM_HIDDEN_LAYERS = 2
HIDDEN_DIM = 1536
DROPOUT_P = 0.3

# --- IRP Config ---
K_ANCHORS = 5000
IRP_OMEGA = 8
IRP_DELTA = 0.7
IRP_RIDGE = 1e-4

# --- Data Cleaning ---
NOISE_THRESHOLD = 0.50
CLEAN_DATA = True 

# --- Config from Notebook 2 (Direct Models) ---
USE_COCO_DATASET = False
MY_DATA_PATH = "/kaggle/input/aml-coco-dataset-10/coco_embeddings_10pct.npz"

# Training HParams for ModernSwiGLU
MODERN_HPARAMS = {
    'layers': 5, 
    'hidden_dim': 1536, 
    'lr': 0.000879, 
    'weight_decay': 0.000178,
    'drop_path': 0.252, 
    'batch_size': 512, 
    'epochs': 500
}
MODERN_SEEDS = [42, 123, 999]