from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='age-gender-detection',
    version='1.0.0',
    description='Age and Gender Detection using Deep Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/age-gender-detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
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
