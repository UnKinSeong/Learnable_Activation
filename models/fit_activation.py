import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    return torch.clamp(raw_output, min=0.0)

def fit_activation_parameters(adaptive_activation, device='cuda', num_epochs=10000, continue_if_not_converged=False):
    """
    Fit parameters a, b, c, d, e, f to match the adaptive activation function
    
    Args:
        adaptive_activation: Trained AdaptiveActivation module
        device: Device to run computations on
        num_epochs: Number of optimization steps
    """
    # Generate input points for fitting
    x = torch.linspace(-35, 35, 1000000, device=device)
    
    # Get the actual output from adaptive activation
    with torch.no_grad():
        y_true = adaptive_activation(x)
    
    # Initialize parameters with smaller values to work better with sigmoid
    a = nn.Parameter(torch.tensor(0.5, device=device))    # positive slope
    b = nn.Parameter(torch.tensor(0.1, device=device))    # negative slope
    c = nn.Parameter(torch.tensor(0.05, device=device))   # positive curvature
    d = nn.Parameter(torch.tensor(0.02, device=device))   # negative curvature
    e = nn.Parameter(torch.tensor(-0.3, device=device))   # offset
    f = nn.Parameter(torch.tensor(2.0, device=device))    # smoothness
    
    optimizer = optim.Adam([a, b, c, d, e, f], lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    best_params = None

    prev_loss = float('inf')
    epoch = 0
    patience = 5  # Number of epochs to wait before early stopping
    no_improve_count = 0
    
    while True:
        optimizer.zero_grad()
        
        # Forward pass with our parametric function
        y_pred = parametric_activation(x, a, b, c, d, e, f)
        
        # Compute loss
        loss = loss_fn(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = (a.item(), b.item(), c.item(), d.item(), e.item(), f.item())
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Check for convergence
        if loss.item() >= prev_loss:
            if not continue_if_not_converged:
                break
        
        # Early stopping check
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch} - No improvement for {patience} epochs")
            break
            
        # Stop if reached max epochs
        epoch += 1
        if epoch >= num_epochs:
            break
            
        prev_loss = loss.item()
    
    return best_params