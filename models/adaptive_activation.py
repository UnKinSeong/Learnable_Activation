import torch
import torch.nn as nn


class AdaptiveActivation(nn.Module):
    def __init__(self, activation_functions=[nn.ReLU, nn.LeakyReLU, nn.SiLU], device='cuda'):
        super(AdaptiveActivation, self).__init__()
        self.weights = nn.Parameter(torch.ones(
            len(activation_functions)).to(device))
        self.bias = nn.Parameter(torch.zeros(
            len(activation_functions)).to(device))
        self.activation_functions = [act().to(device)
                                     for act in activation_functions]

    def forward(self, x):
        weights_softmax = torch.softmax(self.weights, dim=0)
        result = weights_softmax[0] * \
            (self.activation_functions[0](x) + self.bias[0])

        for i in range(1, len(self.activation_functions)):
            result += weights_softmax[i] * \
                (self.activation_functions[i](x) + self.bias[i])

        return result

class ParametricActivation(nn.Module):
    def __init__(self, optimal_parameters):
        super(ParametricActivation, self).__init__()
        self.a, self.b, self.c, self.d, self.e, self.f = optimal_parameters

    def forward(self, x):
        return parametric_activation(x, self.a, self.b, self.c, self.d, self.e, self.f)



# optimal_parameters = [0.7465469837188721, 0.18575870990753174, 0.24590235948562622, -0.024827823042869568, -0.5304253697395325, 2.4235410690307617]
def parametric_activation(x, a, b, c, d, e, f):
    """
    Normalized activation function that maps outputs between 0 and 1
    Parameters remain the same but output is scaled and shifted
    """
    positive_part = a * x + c * torch.pow(x, 2)
    negative_part = b * x + d * torch.pow(x, 2) + e
    
    # Smooth transition using sigmoid
    sigmoid_weight = torch.sigmoid(f * x)
    raw_output = sigmoid_weight * positive_part + (1 - sigmoid_weight) * negative_part
    
    # Normalize output to [0,1] using sigmoid
    return raw_output
    # return positive_part + negative_part