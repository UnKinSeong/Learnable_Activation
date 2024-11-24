import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation import evaluate_dataset
from utils.visualization import visualize_reconstructions, log_activation_stats
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None


def train_model(model, train_loader, test_loaders, num_epochs, device, root_path, encoder_name):
    writer = SummaryWriter(os.path.join(root_path, 'logs', f'{encoder_name}_training'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    checkpoint_filename = os.path.join(root_path, 'checkpoints', f'{encoder_name}_checkpoint.pth')
    start_epoch, saved_loss = load_checkpoint(model, optimizer, checkpoint_filename)

    print(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, num_epochs):
        # Training phase
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

            writer.add_scalar('Loss/batch', loss.item(),
                            epoch * len(train_loader) + batch_idx)

        log_activation_stats(model, writer, epoch)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        save_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_filename)

        # Evaluation phase
        model.eval()
        results = {}

        for dataset_name, loader in test_loaders.items():
            results[dataset_name] = evaluate_dataset(
                model, loader, dataset_name, device,
                visualize_fn=lambda orig, recon, name: visualize_reconstructions(
                    orig, recon, name, writer=writer, step=epoch
                )
            )

            for metric, value in results[dataset_name].items():
                writer.add_scalar(
                    f'Evaluation/{dataset_name}/{metric}', value, epoch)

        # Log evaluation results
        with open(os.path.join(root_path, 'logs', 'evaluation_results.txt'), 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")

            for dataset_name, metrics in results.items():
                f.write(f"\n{dataset_name} Dataset:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")

    writer.close()
