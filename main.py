import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

from models import Autoencoder
from data.loader import load_train_datasets, load_test_datasets
from utils.training import save_checkpoint, load_checkpoint
from utils.evaluation import evaluate_dataset
from utils.visualization import visualize_reconstructions, log_activation_stats


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter('outputs/logs/autoencoder_training')

    # Load training data
    train_loader = load_train_datasets()

    # Create model, criterion, and optimizer
    model = Autoencoder(input_channels=3, device=device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Training parameters
    num_epochs = 400
    checkpoint_filename = 'outputs/checkpoints/autoencoder_checkpoint.pth'

    # Load checkpoint if exists
    start_epoch, saved_loss = load_checkpoint(
        model, optimizer, checkpoint_filename)

    # Load test datasets
    test_loaders = load_test_datasets()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log batch loss
            writer.add_scalar('Loss/batch', loss.item(),
                              epoch * len(train_loader) + batch_idx)

        
        log_activation_stats(model, writer, epoch)

        avg_loss = total_loss / len(train_loader)
        # Log epoch loss
        writer.add_scalar('Loss/epoch', avg_loss, epoch)

        if (epoch+1) % 5 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1,
                        avg_loss, checkpoint_filename)

        model.eval()
        results = {}

        # Evaluate on all test datasets
        for dataset_name, loader in test_loaders.items():
            results[dataset_name] = evaluate_dataset(
                model, loader, dataset_name, device,
                visualize_fn=lambda orig, recon, name: visualize_reconstructions(
                    orig, recon, name, writer=writer, step=epoch
                )
            )

            # Log evaluation metrics to TensorBoard
            for metric, value in results[dataset_name].items():
                writer.add_scalar(
                    f'Evaluation/{dataset_name}/{metric}', value, epoch)

        # Save metrics to file
        with open('outputs/logs/evaluation_results.txt', 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")

            for dataset_name, metrics in results.items():
                f.write(f"\n{dataset_name} Dataset:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
