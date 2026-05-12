"""
Utilities Module
Helper functions for visualization, metrics, and evaluation
Original utility implementations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation and metrics calculation
    Original evaluation methodology
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, task_name='Task'):
        """
        Calculate comprehensive evaluation metrics
        Original metrics calculation
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            task_name (str): Name of task for logging
            
        Returns:
            dict: Calculated metrics
        """
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred
        
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        metrics = {
            'accuracy': accuracy_score(y_true_labels, y_pred_labels),
            'precision': precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'f1': f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        }
        
        self.metrics[task_name] = metrics
        logger.info(f"{task_name} Metrics: {metrics}")
        
        return metrics
    
    def print_classification_report(self, y_true, y_pred, task_name='Task'):
        """
        Print detailed classification report
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            task_name (str): Name of task
        """
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred
        
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        print(f"\n{'='*60}")
        print(f"Classification Report - {task_name}")
        print(f"{'='*60}")
        print(classification_report(y_true_labels, y_pred_labels))


class Visualizer:
    """
    Visualization utilities for training results and analysis
    Original visualization functions
    """
    
    @staticmethod
    def plot_training_history(history, metric='loss', save_path=None):
        """
        Plot training and validation history
        
        Args:
            history: Training history from model.fit()
            metric (str): Metric to plot ('loss' or 'accuracy')
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        if metric == 'loss':
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
        else:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy Over Epochs')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            labels (list): Class labels
            save_path (str): Path to save figure
        """
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred
        
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(images, age_pred, gender_pred, 
                               age_labels=None, gender_labels=None, num_samples=9):
        """
        Plot sample predictions on images
        
        Args:
            images (np.array): Input images
            age_pred (np.array): Age predictions
            gender_pred (np.array): Gender predictions
            age_labels (list): Age group labels
            gender_labels (list): Gender labels
            num_samples (int): Number of samples to display
        """
        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = images[i]
            
            # Denormalize image if normalized
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            
            age_class = np.argmax(age_pred[i])
            gender_class = np.argmax(gender_pred[i])
            age_conf = age_pred[i][age_class]
            gender_conf = gender_pred[i][gender_class]
            
            age_label = f"Age: {age_class}" if age_labels is None else f"Age: {age_labels[age_class]}"
            gender_label = f"Gender: {gender_class}" if gender_labels is None else f"Gender: {gender_labels[gender_class]}"
            
            axes[i].imshow(img)
            axes[i].set_title(f"{age_label} ({age_conf:.2f})\n{gender_label} ({gender_conf:.2f})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


class Logger:
    """Custom logging utilities"""
    
    @staticmethod
    def log_metrics(metrics_dict, stage='Validation'):
        """
        Log metrics in formatted way
        
        Args:
            metrics_dict (dict): Dictionary of metrics
            stage (str): Training stage name
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"{stage} Metrics:")
        logger.info(f"{'='*50}")
        for key, value in metrics_dict.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info(f"{'='*50}\n")
