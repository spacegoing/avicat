# Avicat: Deep State-Space Model for Crypto Market Regimes

A deep learning project for unsupervised discovery of cryptocurrency market regimes using a Deep State-Space Model (DSSM).

-----

## Overview

This repository provides a complete pipeline for training and analyzing a DSSM to identify distinct, interpretable states (or "regimes") in the cryptocurrency market. By leveraging a variety of features—including technical indicators, sentiment data, and market microstructure—the model learns to segment time-series data into periods with similar characteristics, such as high volatility, bullish trends, or sideways movement.

-----

## Features

  * **Unsupervised Regime Discovery**: Automatically identifies a configurable number of market states without pre-labeled data.
  * **Deep Learning Architecture**: Implements a DSSM with a GRU-based inference network for robust temporal modeling.
  * **Comprehensive Feature Engineering**: Includes a preprocessing pipeline to generate a rich feature set for analysis.
  * **End-to-End Workflow**: Covers data preprocessing, model training, state inference, and regime analysis.
  * **Modular & Reproducible**: Organized into a structured Python package with a `pyproject.toml` for easy setup and dependency management.

-----

## Project Structure

The project is organized into a modular `avicat` source package and separate directories for data, configurations, and documentation.

```
.
├── avicat/                 # Main source code package
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # DSSM model architecture
│   ├── training/           # Training and main execution logic
│   ├── analysis/           # Post-training analysis and visualization
│   └── inference/          # Scripts for running inference
├── configs/                # Hydra configuration files
├── data/                   # Project data (raw and processed)
├── docs/                   # Documentation
├── .gitignore
└── pyproject.toml          # Project metadata and dependencies
```

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the project in editable mode:**
    This command reads the `pyproject.toml` file, installs all required dependencies, and makes the `avicat` package and its scripts available in your environment.

    ```bash
    pip install -e .
    ```

-----

## Usage

All commands should be run from the root of the project directory.

### 1\. Preprocess Data

Before training, you must process the raw market data into sequences the model can consume.

```bash
python -m avicat.data.preprocess
```

This script will:

  * Load raw data from `data/raw/`.
  * Engineer features and normalize them.
  * Create time-series sequences and save them to `data/processed/`.

### 2\. Train the Model

To start the training process, use the `avitrain` command.

```bash
avitrain
```

  * This command runs `avicat/training/main.py`.
  * It loads the processed data and the configuration from `configs/crypto_config.yaml`.
  * Checkpoints are saved to the `checkpoints/` directory, and logs are generated.

To run with the debugger:

```bash
python -m pdb -m avicat.training.main
```

### 3\. Run Inference

After training, you can use a saved checkpoint to infer market states on your data.

```bash
aviinfer --checkpoint checkpoints/best_model.pt
```

  * This runs the main inference script (`avicat/inference/infer.py`).
  * It loads the specified checkpoint and the full dataset.
  * Saves a `inferred_states_full.csv` and visualization plots to the `plots/` directory.

### 4\. Analyze a Checkpoint

To perform an in-depth analysis of a specific model checkpoint:

```bash
avianalyze checkpoints/best_model.pt
```

This will:

  * Load the model from the checkpoint.
  * Analyze the discovered market regimes on the validation set.
  * Generate and save a regime transition matrix and feature importance plots to `analysis_results/<checkpoint_name>/`.
