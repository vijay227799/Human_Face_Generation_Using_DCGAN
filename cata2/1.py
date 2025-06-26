import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data(src_dir, target_dir, val_size=0.2):
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory '{src_dir}' does not exist.")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Create train and val directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')

    # Check if directories exist and create them if not
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Get list of all files in the source directory
    all_files = os.listdir(src_dir)
    
    if not all_files:
        raise FileNotFoundError(f"No files found in the source directory '{src_dir}'.")
    
    # Split files into training and validation sets
    train_files, val_files = train_test_split(all_files, test_size=val_size, random_state=42)
    
    # Copy files to the appropriate directories
    for file_name in train_files:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(train_dir, file_name)
        if os.path.isfile(src_file):
            try:
                shutil.copy2(src_file, dest_file)
            except PermissionError as e:
                print(f"Permission error while copying file '{src_file}': {e}")
        else:
            print(f"Skipping non-file item: {src_file}")
    
    for file_name in val_files:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(val_dir, file_name)
        if os.path.isfile(src_file):
            try:
                shutil.copy2(src_file, dest_file)
            except PermissionError as e:
                print(f"Permission error while copying file '{src_file}': {e}")
        else:
            print(f"Skipping non-file item: {src_file}")
    
    print(f"Data preparation complete. Training and validation images are in '{target_dir}'.")

# Run the data preparation
prepare_data('processed_images', 'processed_images')
