#!/usr/bin/env python3
"""
Convert trained Keras model to TensorFlow Lite format.
Usage:
  python convert_to_tflite.py --input models/best_model.h5 --output cat_dog_classifier.tflite [--quantize]
"""
import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained Keras model to TensorFlow Lite format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the trained Keras model (HDF5 or SavedModel folder)"
    )
    parser.add_argument(
        "--output", "-o",
        default="model.tflite",
        help="Path to save the converted .tflite file"
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Enable default optimizations (quantization) for smaller model size"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load Keras model
    print(f"Loading model from: {args.input}")
    model = tf.keras.models.load_model(args.input)
    print("Model loaded successfully.")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        print("Enabling default optimizations (quantization)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    print("Converting to TensorFlow Lite...")
    tflite_model = converter.convert()

    # Save to file
    with open(args.output, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {args.output}")


if __name__ == '__main__':
    main()
