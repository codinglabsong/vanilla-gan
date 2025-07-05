import torch.nn as nn

class Generator(nn.Module):
    """
    Vanilla GAN Generator
    """
    def __init__(self,):
        super().__init__()
        self.layers = nn.Sequential(
            # First upsampling
            nn.Linear(NOISE_DIMENSION, 128, bias=False),
            nn.BatchNorm1d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            # Second upsampling
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            # Third upsampling
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            # Final upsampling
            nn.Linear(512, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
    
    
class Discriminator(nn.Module):
    """
    Vanilla GAN Discriminator
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE, 1024),
            nn.LeakyReLU(0.25),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.layers(x)