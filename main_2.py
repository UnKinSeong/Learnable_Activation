from models import Autoencoder, fit_activation_parameters, parametric_activation
from utils.training import load_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(input_channels=3, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

checkpoint_filename = 'outputs/checkpoints/autoencoder_checkpoint.pth'
start_epoch, saved_loss = load_checkpoint(
    model, optimizer, checkpoint_filename)


activation_parameters = []

# activation = model.adaptive_activation  # Get the specific activation you want to fit
# a, b, c, d, e, f = fit_activation_parameters(activation, num_epochs=100000)
# activation_parameters.append([a, b, c, d, e, f])

# average the parameters
# averaged_parameters = [sum(param[i] for param in activation_parameters) / len(activation_parameters) for i in range(6)]

# After getting the averaged parameters, verify the output range
x_test = torch.linspace(-1e5, 1e5, 100000000, device=device)
# with torch.no_grad():
#     y_fitted = parametric_activation(x_test, *averaged_parameters)
#     print(f"Min output: {y_fitted.min().item():.4f}")
#     print(f"Max output: {y_fitted.max().item():.4f}")

# Modified plotting code to show the [0,1] range clearly
x_test = torch.linspace(-10, 10, 100000, device=device)
averaged_parameters = [0.7465469837188721, 0.18575870990753174, 0.24590235948562622, -0.024827823042869568, -0.5304253697395325, 2.4235410690307617]
plt.figure(figsize=(10, 6))
with torch.no_grad():
    y_fitted = parametric_activation(x_test, *averaged_parameters)
    y_sigmoid = torch.sigmoid(x_test)  # Calculate sigmoid for comparison
    
    plt.plot(x_test.cpu(), y_fitted.cpu(), '--', label='Fitted Function')
    plt.plot(x_test.cpu(), y_sigmoid.cpu(), '-', label='Sigmoid Function')  # Plot sigmoid
    plt.plot(x_test.cpu(), nn.ReLU()(x_test).cpu(), '--', label='ReLU Function')  # Plot relu

    plt.axhline(y=0, color='r', linestyle=':')
    plt.axhline(y=1, color='r', linestyle=':')
    plt.ylim(-0.1, 1.1)  # Set y-axis limits to clearly show the [0,1] range
    plt.legend()
    plt.grid(True)
    plt.title('Fitted Activation (Normalized to [0,1]) and Sigmoid Comparison')
    plt.show()

print(averaged_parameters)
