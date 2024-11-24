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

    def __init__(self, input_channels=1, activation=None, parametric_activation=False, device='cuda', no_encoder=3):
        super(Autoencoder, self).__init__()

        activation_functions = [
            nn.ReLU,
            nn.LeakyReLU,
            nn.Hardswish,
            nn.Hardshrink,
            nn.RReLU,
            nn.ELU,
            nn.SiLU,
            nn.GELU,
            nn.Mish,
            nn.SELU,
            nn.Softshrink,
            nn.Tanhshrink,
            nn.Softplus,
            nn.PReLU
        ]

        if activation is not None:
            if parametric_activation:
                from .fit_activation import fit_activation_parameters
                parameters = fit_activation_parameters(activation, num_epochs=100000)
                self.adaptive_activation = ParametricActivation(parameters).to(device)
            else:
                self.adaptive_activation = activation.to(device)
        else:
            self.adaptive_activation = AdaptiveActivation(
                activation_functions=activation_functions,
                device=device
            )

        self.encoders = nn.ModuleList([
            self.construct_encoder(
                input_channels, self.adaptive_activation, self.adaptive_activation)
            for _ in range(no_encoder)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self.adaptive_activation,
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self.adaptive_activation,
            nn.Conv2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            self.adaptive_activation,
        )

    def forward(self, x):
        encoded = [encoder(x) for encoder in self.encoders]
        encoded = torch.stack(encoded).mean(dim=0)
        # print(encoded.shape)
        # Encoder Shape: (64, 16, 8, 8)
        decoded = self.decoder(encoded)
        return decoded



class AutoencoderParametric(nn.Module):
    def __init__(self, input_channels=1, parameters=None, device='cuda'):
        super(AutoencoderParametric, self).__init__()

        if parameters is None:
            raise ValueError("Parameters are not provided")
        
        self.parametric_activation = ParametricActivation(parameters).to(device)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            self.parametric_activation,
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            self.parametric_activation
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.parametric_activation,
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.parametric_activation,
            nn.Conv2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            self.parametric_activation,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded