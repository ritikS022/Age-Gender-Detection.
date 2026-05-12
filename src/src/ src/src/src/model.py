"""
Model Architecture Module
Defines neural network architectures for age and gender detection
Original model design and architecture implementation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgeGenderModel:
    """
    Age and Gender Detection Model
    Custom architecture combining shared and task-specific layers
    """
    
    def __init__(self, image_size=(224, 224), num_age_groups=8, num_genders=2, 
                 base_model_name='resnet50', pretrained=True):
        """
        Initialize Age Gender Model
        
        Args:
            image_size (tuple): Input image dimensions
            num_age_groups (int): Number of age classification groups
            num_genders (int): Number of gender classes
            base_model_name (str): Base model type ('resnet50' or 'vgg16')
            pretrained (bool): Use pretrained weights
        """
        self.image_size = image_size
        self.num_age_groups = num_age_groups
        self.num_genders = num_genders
        self.base_model_name = base_model_name
        self.model = None
        self.history = None
        
        logger.info(f"Initializing {base_model_name} model")
        self._build_model(pretrained)
    
    def _get_base_model(self, pretrained=True):
        """
        Load pretrained base model
        
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            
        Returns:
            keras.Model: Base model
        """
        weights = 'imagenet' if pretrained else None
        
        if self.base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights=weights
            )
        elif self.base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                input_shape=(*self.image_size, 3),
                include_top=False,
                weights=weights
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        logger.info(f"Base model {self.base_model_name} loaded and frozen")
        
        return base_model
    
    def _build_model(self, pretrained=True):
        """
        Build complete model with shared and task-specific branches
        Original architecture design
        
        Args:
            pretrained (bool): Use pretrained weights
        """
        # Get base model
        base_model = self._get_base_model(pretrained)
        
        # Input layer
        inputs = layers.Input(shape=(*self.image_size, 3))
        
        # Base model
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layers
        shared = layers.Dense(256, activation='relu')(x)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        shared = layers.Dense(128, activation='relu')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        # Age branch (multi-class classification)
        age_branch = layers.Dense(64, activation='relu', name='age_dense1')(shared)
        age_branch = layers.BatchNormalization()(age_branch)
        age_branch = layers.Dropout(0.2)(age_branch)
        
        age_output = layers.Dense(self.num_age_groups, 
                                 activation='softmax', 
                                 name='age_output')(age_branch)
        
        # Gender branch (binary classification)
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense1')(shared)
        gender_branch = layers.BatchNormalization()(gender_branch)
        gender_branch = layers.Dropout(0.2)(gender_branch)
        
        gender_output = layers.Dense(self.num_genders, 
                                    activation='softmax', 
                                    name='gender_output')(gender_branch)
        
        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs=[age_output, gender_output],
            name='AgeGenderDetection'
        )
        
        logger.info("Model architecture created successfully")
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with custom loss weights
        Original loss configuration
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'age_output': 'categorical_crossentropy',
                'gender_output': 'categorical_crossentropy'
            },
            loss_weights={
                'age_output': 0.5,
                'gender_output': 0.5
            },
            metrics={
                'age_output': ['accuracy'],
                'gender_output': ['accuracy']
            }
        )
        
        logger.info("Model compiled successfully")
    
    def summary(self):
        """Print model architecture summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")
    
    def train(self, X_train, y_age_train, y_gender_train,
              X_val, y_age_val, y_gender_val,
              epochs=50, batch_size=32, callbacks=None):
        """
        Train the model
        Original training pipeline
        
        Args:
            X_train, y_age_train, y_gender_train: Training data
            X_val, y_age_val, y_gender_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            callbacks (list): Keras callbacks
            
        Returns:
            History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call _build_model() first.")
        
        logger.info(f"Starting training for {epochs} epochs")
        
        self.history = self.model.fit(
            X_train,
            {'age_output': y_age_train, 'gender_output': y_gender_train},
            validation_data=(
                X_val,
                {'age_output': y_age_val, 'gender_output': y_gender_val}
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=1
        )
        
        return self.history
    
    def predict(self, images):
        """
        Make predictions on images
        
        Args:
            images (np.array): Input images
            
        Returns:
            tuple: (age_predictions, gender_predictions)
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        age_pred, gender_pred = self.model.predict(images)
        return age_pred, gender_pred
    
    def save(self, model_path='models/age_gender_model.h5'):
        """Save model to disk"""
        if self.model:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning("No model to save")
    
    def load(self, model_path='models/age_gender_model.h5'):
        """Load model from disk"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
      
