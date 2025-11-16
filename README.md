# IRP-ResidualMLP Refiner Project

This project implements a two-stage model (IRP + ResidualMLP) for text-to-image retrieval, refactored from a single notebook into a modular Python repository.

## Project Structure

* `/src/irp_refiner`: Core Python package with all logic.
* `/scripts`: Executable scripts for training and prediction.
* `/notebooks`: Jupyter notebooks for experimentation.

## How to Use

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data:**
    * Place the `train.npz` and `test.clean.npz` files into a `/data` directory (e.g., `/kaggle/input/aml-competition/`).
    * Update the paths in `src/irp_refiner/config.py` if needed.

3.  **Run Training:**
    * This will execute the K-Fold cross-validation and save models to the `checkpoints` directory.
    ```bash
    python scripts/train.py
    ```

4.  **Generate Submission:**
    * To generate a submission using a single fold (e.g., fold 3):
    ```bash
    python scripts/predict.py --fold 3
    ```
    * To generate an ensemble submission using all K folds:
    ```bash
    python scripts/predict.py --ensemble
    ```