#!/usr/bin/env python3
"""
TFLite Inference Script using tflite-runtime
Usage:
  python tflite_predict.py --model cat_dog_classifier.tflite --image path/to/image.jpg
"""
import argparse
import numpy as np
from PIL import Image
import platform
from tensorflow.lite.python.interpreter import Interpreter


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with TFLite model")
    parser.add_argument('--model', '-m', required=True, help='Path to .tflite model file')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    return parser.parse_args()


def load_image(path, target_size=(180, 180)):
    img = Image.open(path).convert('RGB').resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    args = parse_args()

    # Load TFLite model
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Get I/O details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    input_data = load_image(args.image)

    # Set tensor and invoke
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get result
    raw_pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = 'Dog' if raw_pred > 0.5 else 'Cat'
    conf = raw_pred if raw_pred > 0.5 else 1 - raw_pred

    print(f"Prediction: {label} (Confidence: {conf*100:.1f}%)")

if __name__ == '__main__':
    main()
