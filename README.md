# Age and Gender Detection using Python

## Project Overview
This project implements a deep learning-based Age and Gender Detection system using Python. It leverages convolutional neural networks (CNN) to predict the age group and gender of individuals from facial images.

**Domain:** Artificial Intelligence (AI)  
**Duration:** 9 Weeks  
**Technology Stack:** Python, TensorFlow, OpenCV, Deep Learning

## Project Objectives
- Develop a robust age and gender detection model using deep learning
- Achieve high accuracy in real-time facial analysis
- Create a user-friendly interface for predictions
- Document the progress weekly
- Minimize code plagiarism through original implementation

## Key Features
- Real-time age and gender detection from webcam
- Batch image processing
- Pre-trained model support and fine-tuning capability
- REST API for integration
- Comprehensive evaluation metrics
- Weekly progress reports

## Project Structure
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.2',
        'opencv-python>=4.5.3.56',
        'tensorflow>=2.6.0',
        'keras>=2.4.3',
        'scikit-learn>=0.24.2',
        'scikit-image>=0.18.3',
        'Pillow>=8.3.1',
        'requests>=2.26.0',
        'scipy>=1.7.0',
        'Flask>=2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.5b0',
            'flake8>=3.9.0',
        ],
        'gpu': [
            'tensorflow-gpu>=2.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'age-gender-detect=src.inference:main',
        ],
    },
    include_package_data=True,
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/age-gender-detection/issues',
        'Source': 'https://github.com/yourusername/age-gender-detection',
        'Documentation': 'https://github.com/yourusername/age-gender-detection/wiki',
    },
)
