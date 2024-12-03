import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size, seed):
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load training data
    train_loader = DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    # Load test data
    test_loader = DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader