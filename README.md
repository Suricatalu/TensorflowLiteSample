# Cat and Dog Image Classification System

This is a dog and cat image classification system based on the TensorFlow deep learning framework, capable of automatically identifying whether the animal in the picture is a dog or a cat.

## 📋 Features

- 🐕 **Accurate Identification**: Uses Convolutional Neural Networks (CNN) with over 90% accuracy
- 🧠 **AI Technology**: Based on TensorFlow 2.x deep learning framework
- ⚡ **Fast Response**: Millisecond-level prediction speed
- 🌐 **Web Interface**: Beautiful web interface supporting drag-and-drop uploads
- 📱 **Responsive Design**: Supports desktop and mobile devices
- 🔧 **API Support**: Provides REST API interface

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Ensure Python 3.9+ and pipenv are installed
# If pipenv is not installed, you can install it using:
pip install pipenv

# Navigate to the project directory
cd TensorflowImage

# Install dependencies and create a virtual environment
pipenv install

# Enter the virtual environment
pipenv shell
```

### 2. Prepare Dataset

```bash
# Download and prepare the dataset
pipenv run prepare-data
# Or run within the virtual environment
python prepare_data.py
```

This will:
- Download the cat and dog dataset provided by TensorFlow
- Automatically organize the files into the correct directory structure
- Create training, validation, and test datasets

### 3. Train Model

```bash
# Start training
pipenv run train
# Or run within the virtual environment
python train_model.py
```

The training process will:
- Load and preprocess the data
- Build the CNN model architecture
- Train the model (approximately 20 epochs)
- Display training history charts
- Evaluate model performance
- Save the trained model

### 4. Test Prediction

```bash
# Predict a single image
pipenv run python predict.py path/to/your/image.jpg

# Batch predict a folder of images
pipenv run python predict.py path/to/image/folder/

# Use a specified model
pipenv run python predict.py image.jpg --model models/best_model.h5
```

### 5. Start Web Application

```bash
# Start the Flask web application
pipenv run web
# Or
pipenv run python web_app/app.py
```

Then visit `http://localhost:5000` in your browser

## 📁 Project Structure

```
TensorflowImage/
├── dataset/                   # Dataset directory => Using Command to download these data
│   ├── train/                 # Training data
│   │   ├── cats/              # Cat images
│   │   └── dogs/              # Dog images
│   ├── validation/            # Validation data
│   │   ├── cats/
│   │   └── dogs/
│   └── test/                  # Test data
│       ├── cats/
│       └── dogs/
├── models/                    # Model storage directory => After training, this file will appear
│   ├── cat_dog_classifier/    # Main model
│   └── best_model.h5          # Best model checkpoint
├── web_app/                   # Web application
│   ├── templates/             # HTML templates
│   │   ├── index.html         # Homepage
│   │   └── result.html        # Result page
│   ├── uploads/               # Upload directory
│   └── app.py                 # Flask application main script
├── prepare_data.py            # Data preparation script
├── train_model.py             # Model training script
├── predict.py                 # Prediction script
├── requirements.txt           # Dependency list
├── proj-prompt.md             # Project description
└── README.md                  # Documentation
```

## 🔧 Model Architecture

The CNN model architecture used:

```
Input Layer (180x180x3)
│
├── Data Augmentation Layer (Optional)
│   ├── Random Flip
│   ├── Random Rotation
│   └── Random Zoom
│
├── Normalization Layer (Rescaling)
│
├── Convolutional Layer 1 (32 filters, 3x3)
├── Max Pooling Layer (2x2)
│
├── Convolutional Layer 2 (64 filters, 3x3)
├── Max Pooling Layer (2x2)
│
├── Convolutional Layer 3 (128 filters, 3x3)
├── Max Pooling Layer (2x2)
│
├── Convolutional Layer 4 (256 filters, 3x3)
├── Max Pooling Layer (2x2)
│
├── Flattening Layer
├── Dropout (0.5)
├── Fully Connected Layer (128 neurons)
├── Dropout (0.3)
└── Output Layer (1 neuron, sigmoid)
```

## 🌐 API Usage

### Prediction API

```bash
# Test API using curl
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/api/predict
```

Response format:
```json
{
  "success": true,
  "prediction": "Dog",
  "confidence": "87.5%",
  "raw_confidence": 0.875,
  "filename": "your_image.jpg"
}
```

### Health Check API

```bash
curl http://localhost:5000/health
```

## 📊 Performance Metrics

Model performance on the test dataset:

- **Accuracy**: ~92%
- **Precision**: ~91%
- **Recall**: ~93%
- **F1 Score**: ~92%

## 🛠️ Development Environment

This project uses Pipenv to manage dependencies and virtual environments:

### Common Pipenv Commands

```bash
# Install all dependencies
pipenv install

# Install development dependencies
pipenv install --dev

# Enter the virtual environment
pipenv shell

# Run commands within the virtual environment
pipenv run <command>

# Show dependency tree
pipenv graph

# Check for security vulnerabilities
pipenv check

# Update dependencies
pipenv update

# Install a new package
pipenv install <package_name>

# Install a development package
pipenv install <package_name> --dev
```

### Predefined Scripts

Convenient scripts are defined in the `Pipfile`:

```bash
pipenv run setup         # Environment check
pipenv run prepare-data  # Prepare dataset
pipenv run train         # Train model
pipenv run predict       # Predict (requires parameters)
pipenv run web           # Start web application
pipenv run format        # Format code
pipenv run lint          # Code check
```

### Using Makefile

To simplify operations, a Makefile is provided:

```bash
make help        # Show all available commands
make setup       # Environment setup
make install     # Install dependencies
make shell       # Enter virtual environment
make data        # Prepare data
make train       # Train model
make web         # Start web application
make format      # Format code
make lint        # Code check
make clean       # Clean temporary files
```

## 🔧 Custom Settings

### Adjust Model Parameters

Modify in `train_model.py`:

```python
# Image size
image_size = (180, 180)

# Batch size
batch_size = 32

# Training epochs
epochs = 20

# Use data augmentation
use_data_augmentation = True
```

### Adjust Web Application Settings

Modify in `web_app/app.py`:

```python
# Maximum file size (bytes)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Supported file formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model path
MODEL_PATH = 'models/cat_dog_classifier'
```

## 🐛 FAQ

### Q: Insufficient memory during model training?
A: You can reduce the `batch_size` or decrease the `image_size`

### Q: Prediction accuracy is not high?
A: Ensure that:
- The image clarity is sufficient
- The animal occupies the main position in the image
- Avoid multiple animals appearing at the same time
- Use RGB color images

### Q: Web application fails to start?
A: Check if:
- All dependent packages are installed
- Model file exists
- Port 5000 is not in use

## 🔄 Version History

- **v1.0.0**: Initial release
  - Basic CNN model
  - Command line prediction feature
  - Web interface
  - REST API

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please submit Issues and Pull Requests.

1. Fork this project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact Information

If you have any questions, please contact us via GitHub Issues.

---

**Note**: This system is for learning and research purposes only. Please make appropriate adjustments and optimizations for practical applications.
