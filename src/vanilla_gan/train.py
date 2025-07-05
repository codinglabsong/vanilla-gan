import torch
from torch import nn
from vanilla_gan.config import config, UNIQUE_RUN_ID
from vanilla_gan.utils import get_device, print_training_progress, generate_image, save_models, make_directory_for_run
from vanilla_gan.data import generate_noise, prepare_dataset
from vanilla_gan.model import initialize_models


def initialize_loss():
    """ Initialize loss function. """
    return nn.BCELoss()


def initialize_optimizers(generator, discriminator):
    """ Initialize optimizers for Generator and Discriminator """
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=config['OPTIMIZER_LR'], betas=tuple(config['OPTIMIZER_BETAS']))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config['OPTIMIZER_LR'], betas=tuple(config['OPTIMIZER_BETAS']))
    return generator_optimizer, discriminator_optimizer


def efficient_zero_grad(model):
    """
    Apply zero_grad more efficiently
    Source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
    """
    for param in model.parameters():
        param.grad = None
      
  
def forward_and_backward(model, data, loss_function, targets):
    """
    Perform forward and backward pass in a generic way. Returns loss value.
    """
    outputs = model(data)
    error = loss_function(outputs, targets)
    error.backward()
    return error.item()


def perform_train_step(generator, discriminator, real_data, \
    loss_function, generator_optimizer, discriminator_optimizer, device=get_device()):
    """ Perform a single training step. """
    
    # 1. PREPARATION
    # Set real and fake labels.
    real_label, fake_label = 1.0, 0.0
    # Get images on CPU or PGU as configured and available
    # Also set 'actual batch size', which can eb smaller than BATCH_SIZE in some cases.
    # This is because the last batch of an epoch may have less examples than the others.
    real_images = real_data[0].to(device) # retrieve the image of (image, label) from the original MNIST dataset
    actual_batch_size = real_images.size(0)
    label = torch.full((actual_batch_size, 1), real_label, device=device)
    
    # 3. TRAINING THE DISCRIMINATOR
    # Zero the gradients for discriminator
    efficient_zero_grad(discriminator)
    # Forward + backward on real images, reshaped
    real_images = real_images.view(real_images.size(0), -1) # (batch_size, 1 x 24 x 24)
    error_real_images = forward_and_backward(discriminator, real_images, \
        loss_function, label)
    # Forward + backward on generated images
    noise = generate_noise(actual_batch_size, device=device)
    generated_images = generator(noise)
    label.fill_(fake_label)
    error_generated_images = forward_and_backward(discriminator, \
        generated_images.detach(), loss_function, label)
    # Optim for discriminator
    discriminator_optimizer.step()
    
    # 3. TRAINING THE GENERATOR
    # Forward + backward + optim for generator, including zero grad
    efficient_zero_grad(generator)
    label.fill_(real_label)
    error_generator = forward_and_backward(discriminator, generated_images, loss_function, label)
    generator_optimizer.step()
    
    # 4. COMPUTING RESULTS
    # Compute loss values in floats for discriminator, which is joint loss.
    error_discriminator = error_real_images + error_generated_images
    # Return generator and discriminator loss so that it can be printed.
    return error_generator, error_discriminator


def perform_epoch(dataloader, generator, discriminator, loss_function, \
    generator_optimizer, discriminator_optimizer, epoch):
    """ Perform a single epoch. """
    for batch_no, real_data in enumerate(dataloader, 0):
        # Perform training step
        generator_loss_val, discriminator_loss_val = perform_train_step(generator, \
            discriminator, real_data, loss_function, \
            generator_optimizer, discriminator_optimizer)
        # Print statistics and generate iamge after every n-th batch
        if batch_no % config['PRINT_STATS_AFTER_BATCH'] == 0:
            print_training_progress(batch_no, generator_loss_val, discriminator_loss_val)
            generate_image(generator, epoch, batch_no)
        # Save models on epoch completion
        save_models(generator, discriminator, epoch)
        # Clear memory after every epoch
        torch.cuda.empty_cache()
        

def main():
    """ Train the GAN """ 
    print("CONFIG")
    print(config, UNIQUE_RUN_ID)   
    # Speed ups
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    
    # Make directory for unique run
    make_directory_for_run()
    # Set fixed random number seed
    torch.manual_seed(42)
    # Get prepared dataset
    dataloader = prepare_dataset()
    # Initialize models
    generator, discriminator = initialize_models()
    # Intialize loss and optimizers
    loss_function = initialize_loss()
    generator_optimizer, discriminator_optimizer = initialize_optimizers(generator, discriminator)
    # Train the model
    for epoch in range(config['NUM_EPOCHS']):
        print(f"Starting epoch {epoch}...")
        perform_epoch(dataloader, generator, discriminator, loss_function, \
            generator_optimizer, discriminator_optimizer, epoch)
    # Finished :-)
    print(f"Finished unique run {UNIQUE_RUN_ID}")
    
if __name__ == "__main__":
    main()