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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("RECSYS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

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

# Небольшой эксперимент для Colab T4
TRAIN_PARAMS = {
    "data_size": 1000,
    "batch_size": 32,
    "num_epochs": 5,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "seed": 42,
    "plot_every": 50,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
