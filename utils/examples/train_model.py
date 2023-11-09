import wandb

from src.training_workflow import train
from utils.config_files.training_config import create_training_config

if __name__ == "__main__":
    wandb.login()
    run_config = create_training_config()
    run = train(config=run_config)
