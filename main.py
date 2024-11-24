import torch
import os
from models import AutoencoderParametric
from data.loader import load_train_datasets_autoencoder, load_test_datasets_autoencoder
from utils.training import train_model

# Define a root path for outputs
# pre_activation_exists = os.path.exists('pre_act/activation.pth')

def auto_encoder_train():
    parameters = (0.13440139591693878, -0.09541631489992142, 1.0021055936813354, -0.053388383239507675, -0.046685654670000076, 69.15345001220703)
    torch.manual_seed(42)
    root_path = 'outputs'
    encoder_name = 'autoencoder_relu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.join(root_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'logs'), exist_ok=True)


    train_loader = load_train_datasets_autoencoder()
    test_loaders = load_test_datasets_autoencoder()

    model = AutoencoderParametric(input_channels=3, parameters=parameters, device=device).to(device)

    num_epochs = 100
    train_model(model, train_loader, test_loaders, num_epochs, device, root_path, encoder_name)

if __name__ == "__main__":
    auto_encoder_train()
