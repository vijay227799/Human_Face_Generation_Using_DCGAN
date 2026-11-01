import os
import shutil
import random

# Define paths
original_folder = 'C:/humans/dataset/Datasets/Celeba_Datasets/img_align_celeba'
organized_folder = 'C:/humans/dataset/Datasets/Celeba_Organized'

# Define countries and genders
countries = ["USA", "UK", "France", "Denmark", "India", "Australia", "Canada"]
genders = ["Male", "Female"]

# Create directory structure
for country in countries:
    for gender in genders:
        os.makedirs(os.path.join(organized_folder, country, gender), exist_ok=True)

# Get list of all images in the original folder
images = os.listdir(original_folder)

# Randomly assign images to countries and genders
for image_name in images:
    country = random.choice(countries)
    gender = random.choice(genders)

    src = os.path.join(original_folder, image_name)
    dst = os.path.join(organized_folder, country, gender, image_name)
    
    shutil.move(src, dst)

print("Dataset has been randomly organized.")
