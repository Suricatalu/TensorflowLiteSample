#!/usr/bin/env python3
"""
Data download and preparation script
Download Kaggle Cats and Dogs dataset and organize file structure
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import tensorflow as tf

def download_sample_dataset():
    """Download sample dataset"""
    print("Downloading sample dataset...")
    
    # Use TensorFlow's sample dataset
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    
    return PATH

def organize_dataset(source_path):
    """Organize dataset into the correct directory structure"""
    base_dir = Path("dataset")
    
    # Create directory structure
    train_dir = base_dir / "train"
    validation_dir = base_dir / "validation"
    test_dir = base_dir / "test"
    
    # Copy files from the downloaded dataset
    source_train_cats = Path(source_path) / "train" / "cats"
    source_train_dogs = Path(source_path) / "train" / "dogs"
    source_val_cats = Path(source_path) / "validation" / "cats"
    source_val_dogs = Path(source_path) / "validation" / "dogs"
    
    if source_train_cats.exists():
        print("Organizing training data...")
        # Copy training data
        shutil.copytree(source_train_cats, train_dir / "cats", dirs_exist_ok=True)
        shutil.copytree(source_train_dogs, train_dir / "dogs", dirs_exist_ok=True)
        
        print("Organizing validation data...")
        # Copy validation data
        shutil.copytree(source_val_cats, validation_dir / "cats", dirs_exist_ok=True)
        shutil.copytree(source_val_dogs, validation_dir / "dogs", dirs_exist_ok=True)
        
        # Move some validation data to test data
        print("Preparing test data...")
        val_cats = list((validation_dir / "cats").glob("*.jpg"))
        val_dogs = list((validation_dir / "dogs").glob("*.jpg"))
        
        # Move the first 100 cat and dog images to the test set
        for i, img in enumerate(val_cats[:100]):
            shutil.move(str(img), test_dir / "cats" / img.name)
        
        for i, img in enumerate(val_dogs[:100]):
            shutil.move(str(img), test_dir / "dogs" / img.name)
            
        print("Dataset preparation complete!")
        print_dataset_info()
    else:
        print(f"Source path not found: {source_path}")

def print_dataset_info():
    """Print dataset information"""
    base_dir = Path("dataset")
    
    for split in ["train", "validation", "test"]:
        split_dir = base_dir / split
        if split_dir.exists():
            cats_count = len(list((split_dir / "cats").glob("*.jpg")))
            dogs_count = len(list((split_dir / "dogs").glob("*.jpg")))
            print(f"{split}: Cats {cats_count}, Dogs {dogs_count}")

if __name__ == "__main__":
    print("Starting preparation of Cats and Dogs image dataset...")
    
    # Download dataset
    source_path = download_sample_dataset()
    
    # Organize dataset
    organize_dataset(source_path)
    
    print("Data preparation complete!")
