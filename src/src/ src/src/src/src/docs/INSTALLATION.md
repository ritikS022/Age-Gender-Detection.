# Installation Guide - Age and Gender Detection

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ritikS022/age-gender-detection.git
cd age-gender-detection
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Dataset (Optional)
- Download UTKFace or similar dataset
- Extract to `data/raw/` directory
- Create CSV with columns: `image, age, gender`

### 6. Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## Troubleshooting

### Issue: TensorFlow GPU not detected
```bash
# Install GPU support (CUDA required)
pip install tensorflow[and-cuda]
```

### Issue: OpenCV import error
```bash
# Reinstall OpenCV
pip install --upgrade opencv-python
```

### Issue: Memory error during training
- Reduce batch_size in `config/config.yaml`
- Use smaller image_size (e.g., 160x160 instead of 224x224)

## Optional: Jupyter Notebooks
```bash
jupyter notebook
# Open notebooks in notebooks/ directory
```

## Next Steps
See [USAGE.md](USAGE.md) for how to train and use the model.
