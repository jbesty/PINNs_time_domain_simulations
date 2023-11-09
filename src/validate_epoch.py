import torch
import wandb

from .loss_functions import loss_function


def validate_epoch(network, dataset):
    network.eval()
    data, target = dataset.get_validation_data()

    # ➡ Forward pass
    with torch.no_grad():
        prediction = network(data)
        loss = loss_function(
            prediction=prediction,
            target=target,
            loss_weights=1 / dataset.std_output_data_training_state_type,
        )

    wandb.log({f"loss_validation": loss}, commit=False)

    return loss
