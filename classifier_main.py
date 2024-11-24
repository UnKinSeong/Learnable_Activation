from data.loader import HAM10000Dataset
from data.transforms import get_transform
import torch
import torch.optim as optim
import os
from torch import nn

from utils.training import load_checkpoint
from models.autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = torch.load('pre_act/activation.pth')

root_path = 'outputs'
encoder_name = 'autoencoder'
model = Autoencoder(input_channels=3, activation=activation, 
                    parametric_activation=True, device=device).to(device)
optimizer = optim.Adam(model.parameters())
checkpoint_filename = os.path.join(root_path, 'checkpoints', f'{encoder_name}_checkpoint.pth')
start_epoch, saved_loss = load_checkpoint(model, optimizer, checkpoint_filename)

# Init dataloaders
transform = get_transform(train=True)
ham10000_dataset_train = HAM10000Dataset(root_path='D:/Learnable_Activation/datasets/HAM10000', transform=transform, split='train')
ham10000_dataset_test = HAM10000Dataset(root_path='D:/Learnable_Activation/datasets/HAM10000', transform=transform, split='test')

ham10000_loader_train = torch.utils.data.DataLoader(ham10000_dataset_train, batch_size=32, shuffle=True)
ham10000_loader_test = torch.utils.data.DataLoader(ham10000_dataset_test, batch_size=32, shuffle=True)

class AttentionClassifier(nn.Module):
    def __init__(self, encoders, num_classes, device):
        super(AttentionClassifier, self).__init__()
        self.encoders = encoders
        
        # Add conv layers for processing encoded features
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Changed input channels to 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers
        self.fc1 = nn.Linear(64, 256)  # Changed input size to match conv2 output
        self.fc2 = nn.Linear(256, num_classes)
        
        self.device = device

    def forward(self, x):
        # Get encoded features and average them
        encoded = [encoder(x) for encoder in self.encoders]
        encoded = torch.stack(encoded).mean(dim=0)
        
        # Process through conv layers
        x = self.conv1(encoded)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        
        # Global average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        
        return x
    
# Create the classifier with 7 classes from HAM10000
classifier = AttentionClassifier(encoders=model.encoders, num_classes=7, device=device).to(device)
criterion = nn.CrossEntropyLoss()
classifier_optimizer = optim.Adam(classifier.parameters())

def train_epoch(classifier, train_loader, optimizer, criterion):
    classifier.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate(classifier, test_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = classifier(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

# Training loop
num_epochs = 100
best_accuracy = 0.0

for epoch in range(num_epochs):
    total_loss = train_epoch(classifier, ham10000_loader_train, classifier_optimizer, criterion)
    accuracy = evaluate(classifier, ham10000_loader_test)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Total Loss: {total_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    
    if accuracy > best_accuracy:
        print('Saving best model...')
        best_accuracy = accuracy
        torch.save(classifier.state_dict(), 'best_classifier.pth')
    
    print('-' * 50)

print(f'Best Accuracy: {best_accuracy:.4f}')
