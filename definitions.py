import pathlib

import torch

PROJECT_PATH = pathlib.Path(__file__).parent
wandb_entity = "jbest"  # provide your wandb details here
wandb_project = "test"

torch.set_default_dtype(torch.float32)
