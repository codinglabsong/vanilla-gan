import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from config import config

def prepare_dataset():
    """ Prepare dataset through DataLoader """
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    # Prepare MNIST dataset
    dataset = MNIST(data_dir, download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # range changes to [-1, 1] to match tanh in Generator
    ]))
    # Batch and shuffle data with DataLoader
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
    # Return dataset through DataLoader
    return trainloader


def generate_noise(number_of_images = 1, noise_dimension = config['NOISE_DIMENSION'], device=None):
    """ Generate noise for number_of_images images, with a specific noise_dimension """
    return torch.randn(number_of_images, noise_dimension, device=device)