from setuptools import find_packages, setup

setup(
    name="dsl_kit",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "patsy>=0.5.0",
        "statsmodels>=0.14.0",
        "pyarrow>=14.0.1",
    ],
    author="Enan Srivastava",
    author_email="contact@enan.dev",
    description="Design-based Supervised Learning (DSL) Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Enan456/dsl-python",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Statistics",
    ],
    python_requires=">=3.9",
)
