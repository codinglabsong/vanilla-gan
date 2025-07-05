# Vanilla GAN

**Vanilla GAN** is a minimal PyTorch implementation of a Generative Adversarial Network trained on the MNIST digit dataset. The project provides lightweight scripts for data preparation and model training along with a configuration file for customizing hyper‑parameters.

## Motivation
This project was created in order to learn and implement GAN on a simple MNIST dataset. It was very helpful to understand not only how GAN works, but how it to utilize this model through PyTorch. I hope to create more useful and impactful project based on this generative technology in the future.

## Features
- **Configurable Training** – main parameters such as number of epochs, batch size and learning rate are defined in `config.yaml`:
  ```yaml
  NUM_EPOCHS: 50
  NOISE_DIMENSION: 50
  BATCH_SIZE: 128
  TRAIN_ON_GPU: True
  PRINT_STATS_AFTER_BATCH: 50
  OPTIMIZER_LR: 0.0002
  OPTIMIZER_BETAS: [0.5, 0.999]
  GENERATOR_OUTPUT_IMAGE_SHAPE: 784
  ```
- **Automatic Dataset Download** – the `prepare_dataset` function downloads MNIST and returns a `DataLoader` with normalized tensors.
- **Simple Network Architecture** – generator and discriminator are built from fully connected layers.
- **Training Utilities** – images and model checkpoints are saved for each run under `data/runs/UNIQUE_RUN_ID`.
- **Developer Tools** – linting with ruff and black

## Installation
1. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) install development tools and pre‑commit hooks:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   pre-commit run --all-files
   ```
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Training
To start training with the default configuration run:
```bash
bash scripts/run_train.sh
```
This script calls `python -m vanilla_gan.train` which handles dataset preparation, model initialization and epoch loops.
Hyper‑parameters may be adjusted by editing `config.yaml`.

## Repository Structure
- `src/vanilla_gan/` – core modules with data loading, model definitions and training logic.
- `scripts/` – helper script for launching training.
- `notebooks/` – Jupyter notebook used for early experiments.

## Requirements
- Python >= 3.10
- PyTorch
- torchvision
- matplotlib

## License
This project is released under the MIT License.