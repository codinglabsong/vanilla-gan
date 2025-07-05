import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from vanilla_gan.data import generate_noise
from vanilla_gan.config import config, UNIQUE_RUN_ID

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def get_device():
    """ Retrieve device based on settings and availability. """
    return torch.device("cuda:0" if torch.cuda.is_available() and config['TRAIN_ON_GPU'] else "cpu")

def make_directory_for_run():
    """ Make a directory for this training run """
    print(f"Preparing training run {UNIQUE_RUN_ID}")
    if not os.path.exists(f"{data_dir}/runs"):
        os.mkdir(f"{data_dir}/runs")
    os.mkdir(f"{data_dir}/runs/{UNIQUE_RUN_ID}")
    
def generate_image(generator, epoch=0, batch=0, device=get_device()):
    """ Genereate subplots with generated examples. """
    images = []
    noise = generate_noise(config['BATCH_SIZE'], device=device)
    generator.eval()
    images = generator(noise)
    plt.figure(figsize=(10, 10))
    for i in range(16):
        # Get iamge
        image = images[i]
        # Convert image back onto CPU and reshape
        image = image.cpu().detach().numpy()
        image = np.reshape(image, (28, 28))
        # Plot
        plt.subplot(4, 4, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    if not os.path.exists(f"{data_dir}/runs/{UNIQUE_RUN_ID}/images"):
        os.mkdir(f"{data_dir}/runs/{UNIQUE_RUN_ID}/images")
    plt.savefig(f"{data_dir}/runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg")
    
def save_models(generator, discriminator, epoch):
    """ Save models at specific point in time. """
    torch.save(generator.state_dict(), f"{data_dir}/runs/{UNIQUE_RUN_ID}/generator_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{data_dir}/runs/{UNIQUE_RUN_ID}/discriminator_{epoch}.pth")
    
def print_training_progress(batch, generator_loss, discriminator_loss):
    """ Print training progress. """
    print(f"Losses after mini-batch {batch:5d}: generator {generator_loss:e}, discriminator {discriminator_loss:e}")