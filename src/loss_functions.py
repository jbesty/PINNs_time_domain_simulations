import torch


def loss_function_point_wise_weighted(prediction, target, loss_weights):
    # the norm essentially squares the squared error again
    loss_full = (prediction - target) * loss_weights
    loss_point_wise = torch.square(torch.linalg.norm(loss_full, dim=1, keepdim=True))
    return loss_point_wise


def loss_function(prediction, target, loss_weights):
    loss_point_wise = loss_function_point_wise_weighted(
        prediction, target, loss_weights
    )
    loss = torch.mean(loss_point_wise)
    return loss
