#!/usr/bin/env python3
"""
Cat vs. Dog Image Classification Model Training Script
Build a CNN model using TensorFlow/Keras
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CatDogClassifier:
    def __init__(self, image_size=(180, 180), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def load_data(self, data_dir="dataset"):
        """Load training and validation data"""
        print("Loading data...")
        
        train_dir = Path(data_dir) / "train"
        val_dir = Path(data_dir) / "validation"
        
        # Load training data
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            validation_split=None,
            seed=123
        )
        
        # Load validation data
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            validation_split=None,
            seed=123
        )
        
        self.class_names = self.train_ds.class_names
        print(f"Class names: {self.class_names}")
        
        # Optimize data loading performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return self.train_ds, self.val_ds
    
    def create_model(self, use_data_augmentation=True):
        """Create CNN model"""
        print("Creating model...")
        
        model_layers = []
        
        # Data augmentation layer (optional)
        if use_data_augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ])
            model_layers.append(data_augmentation)
        
        # Create base CNN architecture
        model_layers.extend([
            layers.Rescaling(1./255),
            
            # First convolutional block
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Second convolutional block
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Third convolutional block
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Fourth convolutional block
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Flatten and fully connected layers
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model = models.Sequential(model_layers)
        
        # Build the model manually to ensure input_shape is set
        self.model.build(input_shape=(None, *self.image_size, 3))
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Display model summary
        self.model.summary()
        
        return self.model
    
    def train(self, epochs=30):
        """Train the model"""
        print(f"Starting model training for {epochs} epochs...")
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Fit the model
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("Please train the model first!")
            return
        
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs_range = range(len(acc))
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_dir="dataset/test"):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Load test data
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Predictions
        predictions = self.model.predict(test_ds)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_labels = []
        for _, labels in test_ds:
            true_labels.extend(labels.numpy())
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, 
                                  target_names=self.class_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return predictions, predicted_classes, true_labels
    
    def save_model(self, model_path="models/cat_dog_classifier"):
        """Save the model"""
        print(f"Saving model to {model_path}...")
        self.model.save(model_path)
        print("Model saved successfully!")
    
    def predict_image(self, image_path):
        """Predict a single image"""
        img = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        prediction = self.model.predict(img_array, verbose=0)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            result = f"Dog (Confidence: {confidence:.2%})"
        else:
            result = f"Cat (Confidence: {1-confidence:.2%})"
        
        return result, confidence

def main():
    print("Starting Cat vs. Dog image classification training...")
    
    # Create classifier
    classifier = CatDogClassifier()
    
    # Load data
    try:
        train_ds, val_ds = classifier.load_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Please run prepare_data.py first to prepare the dataset")
        return
    
    # Create model
    model = classifier.create_model(use_data_augmentation=True)
    
    # Train model
    history = classifier.train(epochs=30)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    try:
        classifier.evaluate_model()
    except Exception as e:
        print(f"Model evaluation failed: {e}")
    
    # Save model
    classifier.save_model()
    
    print("Training complete!")

if __name__ == "__main__":
    main()
