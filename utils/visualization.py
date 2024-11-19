import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from models.adaptive_activation import AdaptiveActivation


def visualize_reconstructions(original, reconstructed, dataset_name, num_images=8, writer=None, step=None):
    """
    Visualize original and reconstructed images side by side
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        dataset_name: Name of the dataset
        num_images: Number of images to visualize
        writer: TensorBoard SummaryWriter instance
        step: Current training step/epoch
    """
    # Create a figure with two rows: original and reconstructed
    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))

    # Create a combined image for TensorBoard
    if writer is not None:
        combined_images = []

    for i in range(num_images):
        # Original images
        if original.size(1) == 3:  # RGB images
            img_orig = original[i].cpu().permute(1, 2, 0)
            img_recon = reconstructed[i].cpu().permute(1, 2, 0)
        else:  # Grayscale images
            img_orig = original[i].cpu().squeeze()
            img_recon = reconstructed[i].cpu().squeeze()
            # Convert grayscale to RGB for consistent TensorBoard display
            if writer is not None:
                img_orig = torch.stack([img_orig] * 3)
                img_recon = torch.stack([img_recon] * 3)
                img_orig = img_orig.permute(1, 2, 0)
                img_recon = img_recon.permute(1, 2, 0)

        # Convert to numpy and normalize to [0, 1] range
        img_orig = img_orig.numpy()
        img_recon = img_recon.numpy()
        
        # Normalize the images
        img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
        img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())

        axes[0, i].imshow(img_orig, vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        axes[1, i].imshow(img_recon, vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

        # Add to combined images for TensorBoard
        if writer is not None:
            combined_images.append(np.vstack([img_orig, img_recon]))

    plt.tight_layout()
    
    # Create step-specific directory structure
    step_str = f'step_{step}' if step is not None else 'final'
    save_dir = os.path.join('outputs', 'images', dataset_name.lower(), step_str)
    # os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure with step information
    # save_path = os.path.join(save_dir, 'reconstructions.png')
    # plt.savefig(save_path)
    plt.close()

    # Log to TensorBoard if writer is provided
    if writer is not None:
        # Combine all images horizontally
        final_image = np.hstack(combined_images)
        writer.add_image(f'Reconstructions/{dataset_name}',
                         final_image,
                         global_step=step,  # This allows tracking over time
                         dataformats='HWC')

def log_activation_stats(model, writer, step):
    """
    Visualize and log adaptive activation functions to TensorBoard
    
    Args:
        model: The autoencoder model
        writer: TensorBoard SummaryWriter instance
        step: Current training step
    """
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveActivation):
            # Create input range for visualization
            x = torch.linspace(-10, 10, 1000, device=module.weights.device)
            
            # Calculate activation output
            with torch.no_grad():
                y = module(x)
                
                # Calculate individual activation components
                components = []
                weights_softmax = torch.softmax(module.weights, dim=0)
                
                # ReLU component
                relu = torch.nn.functional.relu(x + module.bias[0])
                components.append(('ReLU', weights_softmax[0] * relu))
                
                # Leaky ReLU component
                leaky_relu = torch.nn.functional.leaky_relu(x + module.bias[1], negative_slope=0.01)
                components.append(('LeakyReLU', weights_softmax[1] * leaky_relu))
                
                # Hardswish component
                hardswish = torch.nn.functional.hardswish(x + module.bias[2])
                components.append(('Hardswish', weights_softmax[2] * hardswish))

                # Hardshrink component
                hardshrink = torch.nn.functional.hardshrink(x + module.bias[3])
                components.append(('Hardshrink', weights_softmax[3] * hardshrink))

                # RReLU component
                rrelu = torch.nn.functional.rrelu(x + module.bias[4], lower=0.1, upper=0.3)
                components.append(('RReLU', weights_softmax[4] * rrelu))

                # ELU component
                elu = torch.nn.functional.elu(x + module.bias[5], alpha=1.0)
                components.append(('ELU', weights_softmax[5] * elu))
                
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot combined activation function
            ax1.plot(x.cpu().numpy(), y.cpu().numpy(), 'b-', label='Combined')
            ax1.grid(True)
            ax1.legend()
            ax1.set_title(f'Combined Adaptive Activation ({name})')
            ax1.set_xlabel('Input')
            ax1.set_ylabel('Output')
            
            # Plot individual components
            for comp_name, comp_value in components:
                ax2.plot(x.cpu().numpy(), comp_value.cpu().numpy(), label=comp_name)
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Individual Components')
            ax2.set_xlabel('Input')
            ax2.set_ylabel('Output')
            
            plt.tight_layout()
            
            # Convert plot to image for TensorBoard
            fig.canvas.draw()
            
            # Convert matplotlib figure to numpy array
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Log to TensorBoard
            writer.add_image(f'ActivationFunction/{name}', 
                           img.transpose(2, 0, 1),  # Convert to CHW format
                           global_step=step)
            
            plt.close(fig)
