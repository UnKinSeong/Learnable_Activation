import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets
from .transforms import get_transform
import random


class DynamicSamplingDataset(Dataset):
    def __init__(self, datasets, sample_size):
        self.datasets = datasets
        self.sample_size = sample_size
        self.current_indices = self._sample_indices()

    def _sample_indices(self):
        all_indices = []
        for d_idx, dataset in enumerate(self.datasets):
            indices = random.sample(range(len(dataset)), self.sample_size)
            all_indices.extend([(d_idx, s_idx) for s_idx in indices])
        random.shuffle(all_indices)
        return all_indices

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.current_indices[idx]
        return self.datasets[dataset_idx][sample_idx]


def load_train_datasets(sample_size=3000, batch_size=64, path='./datasets'):
    transform = get_transform(train=True)

    # Load datasets
    datasets_list = [
        datasets.MNIST(root=path, train=True,
                       download=True, transform=transform),
        datasets.CIFAR10(root=path, train=True,
                         download=True, transform=transform),
        datasets.Caltech101(root=path, download=True, transform=transform),
        datasets.SVHN(root=path, split='train',
                      download=True, transform=transform),
        datasets.FashionMNIST(root=path, train=True,
                              download=True, transform=transform),
        datasets.KMNIST(root=path, train=True,
                        download=True, transform=transform),
        datasets.STL10(root=path, split='train',
                       download=True, transform=transform)
    ]

    # Create dynamic sampling dataset
    dynamic_dataset = DynamicSamplingDataset(datasets_list, sample_size)
    return DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True)


def load_test_datasets(batch_size=100, path='./datasets'):
    transform = get_transform(train=False)

    # Load test datasets
    test_datasets = {
        'MNIST': datasets.MNIST(root=path, train=False, download=True, transform=transform),
        'CIFAR10': datasets.CIFAR10(root=path, train=False, download=True, transform=transform),
        'Caltech101': datasets.Caltech101(root=path, download=True, transform=transform),
        'SVHN': datasets.SVHN(root=path, split='test', download=True, transform=transform),
        'FashionMNIST': datasets.FashionMNIST(root=path, train=False, download=True, transform=transform),
        'KMNIST': datasets.KMNIST(root=path, train=False, download=True, transform=transform),
        'STL10': datasets.STL10(root=path, split='test', download=True, transform=transform)
    }

    # Create dataloaders
    test_loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for name, dataset in test_datasets.items()
    }

    return test_loaders
