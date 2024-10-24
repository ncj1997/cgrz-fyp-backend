from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import math

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # 3 x 256 x 256
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Dataset class to load only environment images
class EnvironmentDataset(Dataset):
    def __init__(self, env_dir, env_transform=None):
        self.env_dir = env_dir
        self.env_filenames = [x for x in os.listdir(env_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
        self.env_transform = env_transform

    def __len__(self):
        return len(self.env_filenames)

    def __getitem__(self, idx):
        # Load the environment image
        env_filename = self.env_filenames[idx]
        env_path = os.path.join(self.env_dir, env_filename)
        env_image = Image.open(env_path).convert('RGB')

        if self.env_transform:
            env_image = self.env_transform(env_image)

        return env_image

# Save camouflaged image
def save_image(tensor, path):
    unloader = transforms.ToPILImage()  # Converts tensor to PIL image
    image = unloader(tensor)
    image.save(path)


# Function to denormalize the image tensor
def denorm(tensor):
    return (tensor + 1) / 2


# Function to create a collage of camouflaged images
def create_collage(images,folder_id, w=400, h=400, aspect=1.77):
    # Calculate the number of rows and columns based on aspect ratio and number of images
    cols = int(math.sqrt(len(images) * aspect))
    rows = int(math.ceil(float(len(images)) / float(cols)))

    # Shuffle the images randomly
    random.shuffle(images)

    # Set the final collage dimensions
    (width, height) = (w * cols, h * rows)

    # Create a new blank image for the collage
    collage = Image.new("RGB", (width, height))

    # Iterate over the rows and columns to paste images into the collage
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            # If we run out of images, repeat random ones to fill the space
            if i >= len(images):
                i = random.randrange(len(images))
            # Resize the image if necessary to fit the grid
            resized_image = images[i].resize((w, h))
            collage.paste(resized_image, (x * w, y * h))

     # Create a unique folder path for this upload
    folder_path = os.path.join("./static/images/patterns/", folder_id)
    # Make the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    collage_img_path = f"{folder_path}/gan_generated_collage.jpg"
    # Save the final collage image
    collage.save(collage_img_path)

    return collage_img_path


# Define the model directory
MODEL_DIR = './models/'

# Map environment types to model filenames
MODEL_PATHS = {
    'forest': 'forest_camo_generator_model.pth',
    'desert': 'desert_camo_generator_model.pth',
    'snowy': 'snowy_camo_generator_model.pth',
    'urban': 'urban_camo_generator_model.pth'
}

def get_model_path(env_type):
    """
    Given an environment type, returns the corresponding model file path.
    
    :param env_type: The environment type as a string ('forest', 'desert', 'snowy', 'urban')
    :return: Full path to the model file, or None if env_type is invalid
    """
    if env_type not in MODEL_PATHS:
        return " Type is not available"
    
    model_file = MODEL_PATHS[env_type]
    return os.path.join(MODEL_DIR, model_file)

# # Example usage:
# env_type = 'forest'  # This would come from the form or request
# model_path = get_model_path(env_type)

# if model_path:
#     print(f"Model path for {env_type}: {model_path}")
# else:
#     print(f"Invalid environment type: {env_type}")

# Function to generate camouflaged images and create a collage
def generate_camouflage_and_collage(env_folder, env_type,folder_id):
    # Initialize the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)

    # Load the pretrained camouflage generator model
    model_file = get_model_path(env_type)

    print("Selected Model -  ", model_file)

    netG.load_state_dict(torch.load(model_file, map_location=device))
    netG.eval()

    # Define transformations for environment images
    env_transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize to 300x300
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for RGB images
    ])

    # Set up the dataset and dataloader using the provided paths
    dataset = EnvironmentDataset(env_folder, env_transform=env_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    camouflaged_images = []
    # Generate camouflage for a few images
    for i, real_env in enumerate(dataloader):
        if i >= len(dataset):
            break

        real_env = real_env.to(device)

        # Generate camouflage pattern using the generator
        with torch.no_grad():
            camo_pattern = netG(real_env)

        # Convert tensors back to images
        camo_img = denorm(camo_pattern.cpu().squeeze(0))

        # Convert tensor to PIL image
        camo_img_pil = transforms.ToPILImage()(camo_img)

        # Save camouflaged image to the list
        camouflaged_images.append(camo_img_pil)
    print("done model load",len(camouflaged_images))
    # Create the collage from camouflaged images
    generated_collage = create_collage(camouflaged_images,folder_id, w=400, h=400)

    return generated_collage
