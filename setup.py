from setuptools import find_packages, setup

setup(
    name="dsl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "patsy>=0.5.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Double-Supervised Learning (DSL) Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dsl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Statistics",
    ],
    python_requires=">=3.8",
)
