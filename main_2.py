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

activation = model.adaptive_activation
parameters = fit_activation_parameters(activation, num_epochs=100000)

# Plot the results
x_test = torch.linspace(-1e5, 1e5, 1000, device=device)
with torch.no_grad():
    y_fitted = parametric_activation(x_test, *parameters)
    y_original = activation(x_test)
    y_relu = F.relu(x_test)
    y_sigmoid = torch.sigmoid(x_test)
    
    plt.figure(figsize=(12, 12))
    
    # First subplot - Activation comparison
    plt.subplot(2, 2, 1)
    plt.plot(x_test.cpu(), y_fitted.detach().cpu(), label='Fitted')
    plt.plot(x_test.cpu(), y_original.detach().cpu(), '--', label='Original')
    plt.plot(x_test.cpu(), y_relu.cpu(), ':', label='ReLU')
    plt.plot(x_test.cpu(), y_sigmoid.cpu(), '-.', label='Sigmoid')
    plt.grid(True)
    plt.legend()
    plt.title('Activation Function Comparison')

    # Second subplot - Component analysis
    plt.subplot(2, 2, 2)
    left_piece = parameters[0] * x_test + parameters[1]
    right_piece = parameters[2] * x_test + parameters[3]
    plt.plot(x_test.cpu(), left_piece.detach().cpu(), '--', label='Left Piece')
    plt.plot(x_test.cpu(), right_piece.detach().cpu(), '--', label='Right Piece')
    plt.plot(x_test.cpu(), y_fitted.detach().cpu(), label='Combined')
    plt.plot(x_test.cpu(), y_original.detach().cpu(), '--', label='Original')
    plt.axvline(x=parameters[4], color='k', linestyle=':', label='Transition Point')
    plt.grid(True)
    plt.legend()
    plt.title('Component Analysis')

    # Third subplot - Zoomed view of activation functions
    plt.subplot(2, 2, 3)
    x_zoom = torch.linspace(-1e5, 1e5, 1000, device=device)
    y_fitted_zoom = parametric_activation(x_zoom, *parameters)
    y_relu_zoom = F.relu(x_zoom)
    y_tanhshrink_zoom = torch.nn.functional.tanhshrink(x_zoom)
    y_sigmoid_zoom = torch.sigmoid(x_zoom)
    
    plt.plot(x_zoom.cpu(), y_fitted_zoom.detach().cpu(), label='Fitted')
    plt.plot(x_zoom.cpu(), y_relu_zoom.cpu(), ':', label='ReLU')
    plt.plot(x_zoom.cpu(), y_sigmoid_zoom.cpu(), '--', label='Sigmoid')
    plt.plot(x_zoom.cpu(), y_tanhshrink_zoom.cpu(), '-.', label='Tanhshrink')
    plt.grid(True)
    plt.legend()
    plt.title('Zoomed View (-10 to 10)')

    # Fourth subplot - Original activation function
    plt.subplot(2, 2, 4)
    plt.plot(x_test.cpu(), y_original.detach().cpu(), label='Original Activation', color='orange')
    plt.grid(True)
    plt.legend()
    plt.title('Original Activation Function')

plt.tight_layout()
plt.show()

print("Fitted parameters:", parameters)
