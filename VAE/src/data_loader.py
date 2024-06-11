import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_dataloaders(batch_size=100, validation_split=0.1):
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=data_transforms)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=data_transforms)

    num_train = len(training_data)
    num_validation = int(num_train * validation_split)
    num_train -= num_validation

    train_dataset, validation_dataset = random_split(training_data, [num_train, num_validation])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader
