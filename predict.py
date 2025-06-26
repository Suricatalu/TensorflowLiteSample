#!/usr/bin/env python3
"""
Single Image Prediction Script
Load a trained model and predict on new images
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np

class CatDogPredictor:
    def __init__(self, model_path="models/cat_dog_classifier"):
        """Initialize the predictor"""
        self.model_path = model_path
        self.model = None
        self.image_size = (180, 180)
        self.class_names = ['cats', 'dogs']
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
    def preprocess_image(self, image_path):
        """Preprocess the image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and resize the image
        img = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        
        return img_array
    
    def predict(self, image_path, show_confidence=True):
        """Predict a single image"""
        if self.model is None:
            self.load_model()
        
        # Preprocess the image
        img_array = self.preprocess_image(image_path)
        
        # Making prediction
        prediction = self.model.predict(img_array, verbose=0)
        confidence = prediction[0][0]
        
        # Determine the result
        if confidence > 0.5:
            predicted_class = "Dog"
            probability = confidence
        else:
            predicted_class = "Cat" 
            probability = 1 - confidence
        
        result = {
            'class': predicted_class,
            'confidence': probability,
            'raw_prediction': confidence
        }
        
        if show_confidence:
            print(f"Prediction Result: {predicted_class} (Confidence: {probability:.2%})")
        
        return result
    
    def predict_batch(self, image_folder):
        """Batch predict all images in a folder"""
        if self.model is None:
            self.load_model()
        
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Folder not found: {image_folder}")
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in image_folder.iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            print("No supported image files found in the folder")
            return []
        
        results = []
        print(f"Predicting {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files, 1):
            try:
                result = self.predict(str(image_file), show_confidence=False)
                result['filename'] = image_file.name
                results.append(result)
                
                print(f"[{i}/{len(image_files)}] {image_file.name}: "
                      f"{result['class']} ({result['confidence']:.2%})")
                
            except Exception as e:
                print(f"Error during prediction for {image_file.name}: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Cat vs. Dog Image Classification Prediction')
    parser.add_argument('input', help='Input image path or folder path')
    parser.add_argument('--model', default='models/cat_dog_classifier', 
                       help='Model path (default: models/cat_dog_classifier)')
    
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = CatDogPredictor(args.model)
        
        # Check if input is a file or a folder
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single file prediction
            result = predictor.predict(str(input_path))
            
        elif input_path.is_dir():
            # Batch prediction
            results = predictor.predict_batch(str(input_path))
            
            # Prediction statistics
            if results:
                cat_count = sum(1 for r in results if r['class'] == 'Cat')
                dog_count = sum(1 for r in results if r['class'] == 'Dog')
                
                print(f"\nPrediction Statistics:")
                print(f"Cats: {cat_count}")
                print(f"Dogs: {dog_count}")
                print(f"Total: {len(results)}")
        
        else:
            print(f"Error: {args.input} is not a valid file or folder")
            sys.exit(1)
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # If no command line arguments are provided, show usage instructions
    if len(sys.argv) == 1:
        print("Cat vs. Dog Image Prediction Tool")
        print("\nUsage:")
        print("  python predict.py <image_path>         # Predict a single image")
        print("  python predict.py <folder_path>      # Batch prediction")
        print("  python predict.py <path> --model <model_path>  # Use a specific model")
        print("\nExamples:")
        print("  python predict.py test_image.jpg")
        print("  python predict.py dataset/test/cats/")
        print("  python predict.py my_image.png --model models/best_model.h5")
        sys.exit(0)
    
    main()
