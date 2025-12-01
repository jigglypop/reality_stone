import torch
import numpy as np
import random
import os

def get_device() -> str:
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = 42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_mnist_dataloaders(data_dir="./data", batch_size=256, test_batch_size=1000, download=True):
    """Loads MNIST train and test dataloaders with standard normalization."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate_accuracy(model, test_loader, device):
    """Evaluates classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

