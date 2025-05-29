from setuptools import setup, find_packages

setup(
    name="dino-object-detection",
    version="0.1.0",
    description="Real-time object detection using DETR model",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "transformers>=4.30.2",
        "opencv-python>=4.8.1.78",
        "pillow>=9.5.0",
        "numpy>=1.24.3",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.1",
        "pycocotools>=2.0.6",
        "huggingface-hub>=0.16.4",
        "datasets>=2.12.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 