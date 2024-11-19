import torch
import torch.nn as nn
from .adaptive_activation import AdaptiveActivation, ParametricActivation



class Autoencoder(nn.Module):
    def construct_encoder(self, input_channels=1, activation_function1=None, activation_function2=None):
        activation_function1 = activation_function1 if activation_function1 is not None else nn.ReLU()
        activation_function2 = activation_function2 if activation_function2 is not None else nn.ReLU()

        return nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            activation_function1,
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            activation_function2
        )

    def __init__(self, input_channels=1, device='cuda', no_encoder=3):
        super(Autoencoder, self).__init__()

        activation_functions = [
            nn.ReLU,
            nn.LeakyReLU,
            nn.Hardswish,
            nn.Hardshrink,
            nn.RReLU,
            nn.ELU,
        ]

        # self.adaptive_activation = AdaptiveActivation(
        #     activation_functions=activation_functions,
        #     device=device
        # )

        self.parametric_activation = ParametricActivation(
            optimal_parameters=[0.7465469837188721, 0.18575870990753174, 0.24590235948562622, -0.024827823042869568, -0.5304253697395325, 2.4235410690307617]
        )

        self.encoders = nn.ModuleList([
            self.construct_encoder(
                input_channels, self.parametric_activation, self.parametric_activation)
            for _ in range(no_encoder)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self.parametric_activation,
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self.parametric_activation,
            nn.Conv2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = [encoder(x) for encoder in self.encoders]
        encoded = torch.stack(encoded).mean(dim=0)
        decoded = self.decoder(encoded)
        return decoded
