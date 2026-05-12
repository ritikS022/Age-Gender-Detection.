# Usage Guide - Age and Gender Detection

## Quick Start

### 1. Prepare Data
```python
from src.data_loader import DataLoader

# Initialize data loader
loader = DataLoader(image_size=(224, 224))

# Load images
images, ages, genders = loader.load_images(
    image_dir='data/raw/images',
    csv_file='data/raw/labels.csv'
)

# Encode labels
loader.encode_labels()

# Create age bins
loader.create_age_bins(num_bins=8)

# Split data
train_data = loader.split_data()
```

### 2. Train Model
```python
from src.model import AgeGenderModel
from src.data_loader import DataLoader

# Load and prepare data
loader = DataLoader()
X_train, X_val, X_test, y_age_train, y_age_val, y_age_test, \
    y_gender_train, y_gender_val, y_gender_test = loader.split_data()

# Create and compile model
model = AgeGenderModel(
    image_size=(224, 224),
    num_age_groups=8,
    num_genders=2,
    base_model_name='resnet50'
)
model.compile_model(learning_rate=0.001)

# Train
history = model.train(
    X_train, y_age_train, y_gender_train,
    X_val, y_age_val, y_gender_val,
    epochs=50,
    batch_size=32
)

# Save model
model.save('models/age_gender_model.h5')
```

### 3. Make Predictions
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('models/age_gender_model.h5')

# Prepare image
image = cv2.imread('test_image.jpg')
image = cv2.resize(image, (224, 224))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

# Predict
age_pred, gender_pred = model.predict(image)
age_class = np.argmax(age_pred[0])
gender_class = np.argmax(gender_pred[0])

print(f"Predicted Age Group: {age_class}")
print(f"Predicted Gender: {gender_class}")
print(f"Age Confidence: {age_pred[0][age_class]:.2%}")
print(f"Gender Confidence: {gender_pred[0][gender_class]:.2%}")
```

### 4. Evaluate Model
```python
from src.utils import ModelEvaluator, Visualizer

evaluator = ModelEvaluator()

# Calculate metrics
age_metrics = evaluator.calculate_metrics(
    y_age_test, age_pred, 
    task_name='Age Detection'
)

gender_metrics = evaluator.calculate_metrics(
    y_gender_test, gender_pred, 
    task_name='Gender Detection'
)

# Print classification report
evaluator.print_classification_report(y_age_test, age_pred, 'Age Detection')
evaluator.print_classification_report(y_gender_test, gender_pred, 'Gender Detection')

# Visualize results
Visualizer.plot_training_history(history, metric='loss')
Visualizer.plot_confusion_matrix(y_age_test, age_pred)
Visualizer.plot_sample_predictions(X_test, age_pred, gender_pred)
```

## Configuration
Edit `config/config.yaml` to customize:
- Model architecture
- Training parameters
- Data paths
- Evaluation settings

## API Usage
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed API documentation.
