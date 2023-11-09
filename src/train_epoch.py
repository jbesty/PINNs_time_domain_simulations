import torch
import wandb

from .loss_functions import loss_function


def train_epoch_lbfgs(network, dataset, optimiser):
    network.train()

    data, target, target_dt = dataset.get_training_data()

    data_collocation, data_control_input_collocation = dataset.get_collocation_data()

    def closure():
        optimiser.zero_grad()
        prediction, prediction_dt = network.forward_dt(data)

        loss_prediction = loss_function(
            prediction=prediction,
            target=target,
            loss_weights=1 / dataset.std_output_data_training_state_type,
        )

        if network.dt_regulariser > 0.0:
            loss_prediction_dt = loss_function(
                prediction=prediction_dt,
                target=target_dt,
                loss_weights=torch.ones((1, dataset.n_states)),
            )
        else:
            loss_prediction_dt = 0.0

        if network.physics_regulariser > 0.0:
            if len(data) == len(data_collocation):
                # TODO: ATTENTION: only applicable with the very specific approach in the paper - double check!!
                prediction_collocation = prediction
                prediction_collocation_dt = prediction_dt
            else:
                prediction_collocation, prediction_collocation_dt = network.forward_dt(
                    data_collocation
                )
            lhs_collocation, rhs_collocation = dataset.calculate_state_update(
                prediction_collocation,
                prediction_collocation_dt,
                data_control_input_collocation,
            )

            loss_physics = loss_function(
                prediction=lhs_collocation,
                target=rhs_collocation,
                loss_weights=torch.ones((1, dataset.n_states)),
            )
        else:
            loss_physics = 0.0

        loss = (
            loss_prediction
            + network.dt_regulariser * loss_prediction_dt
            + network.physics_regulariser * loss_physics
        )
        # ⬅ Backward pass + weight update
        loss.backward()

        # commit=True to see every optimiser step
        wandb.log(
            {
                "loss_prediction": loss_prediction,
                "loss_prediction_dt": loss_prediction_dt,
                "loss_physics": loss_physics,
            },
            commit=False,
        )
        return loss

    optimiser.step(closure)

    with torch.no_grad():
        prediction, _ = network.forward_dt(data)

        loss_prediction = loss_function(
            prediction=prediction,
            target=target,
            loss_weights=1 / dataset.std_output_data_training_state_type,
        )

    return loss_prediction
