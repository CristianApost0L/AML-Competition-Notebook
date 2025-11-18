# AML Competition


This repository provides two distinct approaches for text-to-image retrieval submissions:

1. **K-Fold Ensemble with IRP + ResidualMLP**: A two-stage model combining an Image-Text Retrieval Pipeline (IRP) with a Residual Multi-Layer Perceptron (ResidualMLP), trained and ensembled using K-Fold cross-validation.
2. **Ensemble of Three Distinct MLP Adapters**: An alternative approach that ensembles predictions from three independently trained Multi-Layer Perceptron (MLP) adapters.

Both methods are modular, and the codebase is refactored from a single notebook into a clean Python project.

## Project Structure

- **`src/`**: Core Python package with all model and utility code
- **`scripts/`**: Command-line scripts for training and prediction
- **`notebooks/`**: Jupyter notebooks

---

For more details, see the code and notebooks in this repository.