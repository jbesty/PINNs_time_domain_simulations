from collections import OrderedDict

import torch


class Standardise(torch.nn.Module):
    """
    Scale the input to the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Standardise, self).__init__()
        self.mean = torch.nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=torch.ones(n_neurons), requires_grad=False
        )
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.mean) / (self.standard_deviation + self.eps)

    def set_standardisation(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception("Input statistics are not 1-D tensors.")

        if (
            not torch.nonzero(self.standard_deviation).shape[0]
            == standard_deviation.shape[0]
        ):
            raise Exception(
                "Standard deviation in standardisation contains elements equal to 0."
            )

        self.mean = torch.nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=standard_deviation, requires_grad=False
        )


class Scale(torch.nn.Module):
    """
    Scale the input to the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Scale, self).__init__()
        self.mean = torch.nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=torch.ones(n_neurons), requires_grad=False
        )
        self.eps = 1e-8

    def forward(self, input):
        return self.mean + input * self.standard_deviation

    def set_scaling(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception("Input statistics are not 1-D tensors.")

        if (
            not torch.nonzero(self.standard_deviation).shape[0]
            == standard_deviation.shape[0]
        ):
            raise Exception(
                "Standard deviation in scaling contains elements equal to 0."
            )

        self.mean = torch.nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=standard_deviation, requires_grad=False
        )


class NeuralNetwork(torch.nn.Module):
    """
    A simple multi-layer perceptron network, with optional input and output standardisation/scaling and the computation
    of output to input sensitivities.
    """

    def __init__(
        self,
        hidden_layer_size: int,
        n_hidden_layers: int,
        n_input_neurons: int,
        n_output_neurons: int,
        pytorch_init_seed: int = None,
    ):
        """
        :param hidden_layer_size: Number of neurons per hidden layer (could be extended to varying size)
        :param n_hidden_layers: Number of hidden layers
        :param pytorch_init_seed:
        """
        super(NeuralNetwork, self).__init__()

        if type(pytorch_init_seed) is int:
            torch.manual_seed(pytorch_init_seed)

        self.epochs_total = 0
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.dt_regulariser = torch.tensor(0.0)
        self.physics_regulariser = torch.tensor(0.0)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size

        neurons_in_layers = (
            [self.n_input_neurons]
            + [hidden_layer_size] * n_hidden_layers
            + [self.n_output_neurons]
        )
        layer_dictionary = OrderedDict()

        layer_dictionary["input_standardisation"] = Standardise(self.n_input_neurons)

        for ii, (neurons_in, neurons_out) in enumerate(
            zip(neurons_in_layers[:-2], neurons_in_layers[1:-1])
        ):
            layer_dictionary[f"dense_{ii}"] = torch.nn.Linear(
                in_features=neurons_in, out_features=neurons_out, bias=True
            )
            layer_dictionary[f"activation_{ii}"] = torch.nn.Tanh()
            torch.nn.init.xavier_normal_(
                layer_dictionary[f"dense_{ii}"].weight,
            )

        layer_dictionary["output_layer"] = torch.nn.Linear(
            in_features=neurons_in_layers[-2],
            out_features=neurons_in_layers[-1],
            bias=True,
        )
        torch.nn.init.xavier_normal_(layer_dictionary["output_layer"].weight, gain=1.0)

        layer_dictionary["output_scaling"] = Scale(self.n_output_neurons)

        self.dense_layers = torch.nn.Sequential(layer_dictionary)

    def standardise_input(self, input_mean, input_standard_deviation):
        self.dense_layers.input_standardisation.set_standardisation(
            mean=input_mean, standard_deviation=input_standard_deviation
        )

    def scale_output(self, output_mean, output_standard_deviation):
        self.dense_layers.output_scaling.set_scaling(
            mean=output_mean, standard_deviation=output_standard_deviation
        )

    def forward(self, x):
        return self.dense_layers(x)

    def forward_dt(self, x):
        input_vector = torch.zeros(x.shape[0], x.shape[1])
        input_vector[:, 0] = 1.0
        prediction, prediction_dt = torch.autograd.functional.jvp(
            self.forward, x, v=input_vector, create_graph=True
        )
        return prediction, prediction_dt

    def predict_dt(self, x):
        self.eval()

        with torch.no_grad():
            input_vector = torch.zeros(x.shape[0], x.shape[1])
            input_vector[:, 0] = 1.0
            prediction, prediction_dt = torch.autograd.functional.jvp(
                self.forward, x, v=input_vector, create_graph=False
            )
        return prediction, prediction_dt
