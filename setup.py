from setuptools import setup, find_packages

setup(
    name="physretinexnet",
    version="1.0.0",
    author="Sonali Chendake",
    description="PhysRetinexNet: Physics-Guided Low-Light Image Enhancement with Illumination-Anchored Reflectance and Strict No-Reference Evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sonali-Chendake/low-light-image-enhancement",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "pillow>=8.0.0",
        "pandas>=1.3.0",
        "lpips>=0.1.4",
        "piq>=0.7.0",
    ],
)
