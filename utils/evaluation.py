import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import torch


def compute_metrics(original, reconstructed):
    """Compute various reconstruction metrics"""
    # Convert tensors to numpy arrays and reshape
    original = original.cpu().numpy().reshape(original.shape[0], -1)
    reconstructed = reconstructed.cpu().numpy().reshape(
        reconstructed.shape[0], -1)

    # Compute MSE
    mse = mean_squared_error(original, reconstructed)

    # For PSNR, reshape back to image format
    original_img = original.reshape(-1, 3, 32, 32)
    reconstructed_img = reconstructed.reshape(-1, 3, 32, 32)

    # Compute PSNR (average over batch)
    psnr = np.mean([
        peak_signal_noise_ratio(orig, recon, data_range=1.0)
        for orig, recon in zip(original_img, reconstructed_img)
    ])

    return {
        'MSE': mse,
        'PSNR': psnr
    }


def evaluate_dataset(model, loader, dataset_name, device, visualize_fn=None):
    total_metrics = {'MSE': 0.0, 'PSNR': 0.0}
    num_batches = 0

    # Get one batch for visualization
    sample_batch = None
    sample_recon = None

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            reconstructed = model(data)

            # Save first batch for visualization
            if batch_idx == 0:
                sample_batch = data[:8].clone()
                sample_recon = reconstructed[:8].clone()

            # Compute metrics
            metrics = compute_metrics(data, reconstructed)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

    # Average metrics
    avg_metrics = {key: value/num_batches for key,
                   value in total_metrics.items()}

    # Visualize reconstructions if function provided
    if visualize_fn is not None:
        visualize_fn(sample_batch, sample_recon, dataset_name)

    return avg_metrics
