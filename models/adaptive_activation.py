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



# optimal_parameters = [0.15180999040603638, -0.14924363791942596, 1.002923846244812, -0.07417581230401993, -0.06182337552309036, 7.587813377380371]
def parametric_activation(x, a1, b1, a2, b2, transition_point, smoothness):
    """
    Piecewise activation function with smooth transition:
    - Left side: a1*x + b1 (for x < transition_point)
    - Right side: a2*x + b2 (for x > transition_point)
    - Smooth transition around transition_point using sigmoid blending
    
    Args:
        x: Input tensor
        a1, b1: Slope and intercept for left piece
        a2, b2: Slope and intercept for right piece
        transition_point: Point where transition occurs
        smoothness: Controls smoothness of transition
    """
    left_piece = a1 * x + b1
    right_piece = a2 * x + b2
    
    # blend = torch.sigmoid(smoothness * (x - transition_point))
    
    # return blend * right_piece + (1 - blend) * left_piece

    # Simple threshold-based connection
    return torch.where(x < transition_point, left_piece, right_piece)