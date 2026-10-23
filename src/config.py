import os
import shutil
import json
from collections import Counter
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from IPython.display import clear_output

import torch
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("RECSYS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def setup_paths() -> None:
    """Настроить пути для локального запуска, Colab или Kaggle."""
    global DATA_DIR, ARTIFACTS_DIR

    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"

    if Path("/kaggle/input").is_dir():
        artifacts_dir = Path("/kaggle/working/artifacts")
        if "RECSYS_DATA_DIR" not in os.environ:
            files = sorted(Path("/kaggle/input").rglob("listens.parquet"))
            if len(files) != 1:
                raise ValueError(
                    "Подключите dataset с одним listens.parquet или задайте "
                    "RECSYS_DATA_DIR — путь к папке с нужным файлом."
                )
            data_dir = files[0].parent
    elif Path("/content").is_dir():
        from google.colab import drive

        drive.mount("/content/drive")
        drive_root = Path("/content/drive/MyDrive/recsys")
        artifacts_dir = drive_root / "artifacts"
        if "RECSYS_DATA_DIR" not in os.environ:
            data_dir = Path("/content/recsys/data")
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_root / "data/listens.parquet", data_dir / "listens.parquet")

    DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", data_dir))
    ARTIFACTS_DIR = Path(os.environ.get("RECSYS_ARTIFACTS_DIR", artifacts_dir))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Параметры модели GPT-2
GPT2_PARAMS = {
    "n_positions": 64,
    "n_ctx": 64,
    "n_embd": 128,
    "n_layer": 2,
    "n_head": 4,
    "bos_token_id": None,
    "eos_token_id": None,
}

# Параметры обучения
TRAIN_PARAMS = {
    "data_size": 1000,
    "min_frequency": 5,
    "batch_size": 32,
    "num_epochs": 5,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "seed": 42,
    "plot_every": 50,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
