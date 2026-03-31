import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import random
from PIL import ImageEnhance
import cv2


def add_noise(image):
    # Add Gaussian noise to an image
    row, col = image.shape
    mean = 0
    sigma = 10  # Standard deviation of the noise
    gauss = np.random.normal(mean, sigma, (row, col)).reshape(row, col)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to be in the range [0, 255]
    return noisy_image.astype(np.uint8)


# Directory where the original  images are stored
original_images_dir = '/lines/'
augmented_images_dir = '/lines/Aug/'

# Create directory if it doesn’t exist
if not os.path.exists(augmented_images_dir):
    os.makedirs(augmented_images_dir)


# Function to apply augmentations
def apply_augmentations(image):
    # Shear transformation
    width, height = image.size
    shear_factor = random.uniform(-0.001, 0.001)  # Small shear factor for minimal distortion
    shear_matrix = (1, shear_factor, 0, shear_factor, 1, 0)
    sheared_image = image.transform((width, height), Image.AFFINE, shear_matrix, resample=Image.BICUBIC)

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(sheared_image)
    brightness_factor = random.uniform(0.9, 1.1)  # Slight brightness change
    bright_image = enhancer.enhance(brightness_factor)

    return bright_image


# Load each image and generate augmented versions
image_files = os.listdir(original_images_dir)
print(len(image_files))
num_augmented_images = 120 // len(image_files)  # Augmented images needed per original image

for image_file in image_files:
    img_path = os.path.join(original_images_dir, image_file)

    # Ensure it is an image file
    if not img_path.lower().endswith(('.tiff','.png', '.jpg', '.jpeg','.tif',)):
        continue

    # Load the original image in grayscale mode
    img = Image.open(img_path)

    # Generate augmented images
    for i in range(num_augmented_images):
        augmented_img = apply_augmentations(img)

        # Construct a unique file name for each augmented image
        save_path = os.path.join(augmented_images_dir, f"{os.path.splitext(image_file)[0]}_aug_{i}.tiff")

        # Save the image
        augmented_img.save(save_path)
