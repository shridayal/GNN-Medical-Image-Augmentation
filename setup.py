from setuptools import setup, find_packages

setup(
    name="gnn-medical-augmentation",
    version="0.1.0",
    description="GNN-Guided Generative AI for Medical Image Augmentation",
    author="Shridayal Yadav",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.3',
        'scikit-image>=0.21.0',
        'matplotlib>=3.7.1',
        'Pillow>=9.5.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
)