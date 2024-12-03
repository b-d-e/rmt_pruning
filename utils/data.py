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

    # Set up CUDA if available
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load training data
    train_loader = DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    # Load test data
    test_loader = DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, test_loader