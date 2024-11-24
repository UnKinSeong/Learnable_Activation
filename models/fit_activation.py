import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def parametric_activation(x, a1, b1, a2, b2, transition_point, smoothness):
    left_piece = a1 * x + b1
    right_piece = a2 * x + b2
    
    blend = torch.sigmoid(smoothness * (x - transition_point))
    
    return blend * right_piece + (1 - blend) * left_piece

def fit_activation_parameters(adaptive_activation, device='cuda', num_epochs=10000):
    x = torch.linspace(-10, 10, 1000, device=device)
    
    with torch.no_grad():
        y_true = adaptive_activation(x)
    
    # Initialize parameters
    a1 = nn.Parameter(torch.tensor(0.1, device=device))
    b1 = nn.Parameter(torch.tensor(-1.0, device=device))
    a2 = nn.Parameter(torch.tensor(1.0, device=device))
    b2 = nn.Parameter(torch.tensor(0.0, device=device))
    t = nn.Parameter(torch.tensor(0.0, device=device))
    s = nn.Parameter(torch.tensor(2.0, device=device))
    
    optimizer = optim.Adam([a1, b1, a2, b2, t, s], lr=0.01)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    best_params = None
    patience = 100
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        y_pred = parametric_activation(x, a1, b1, a2, b2, t, s)
        
        loss = loss_fn(y_pred, y_true)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = (a1.item(), b1.item(), a2.item(), b2.item(), t.item(), s.item())
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch} - No improvement for {patience} epochs")
            break
    
    return best_params

def plot_activation(parameters, x_range=(-10, 10)):
    """
    Plot the fitted activation function
    """
    x = torch.linspace(x_range[0], x_range[1], 1000)
    with torch.no_grad():
        y = parametric_activation(x, *parameters)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.title('Fitted Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()