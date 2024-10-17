# setup.py
from setuptools import setup, find_packages

setup(
    name="PySW",            # Package name
    version="0.9.9",              # Package version
    description="A Python package with classes, utils, and solver functionality",  # Short description
    long_description=open("README.md").read(),  # Detailed description from README
    long_description_content_type="text/markdown",  # Specify markdown if README.md is used
    author="- Giovanni Francesco Diotallevi - Irving Leander Reascos Valencia",           # Author's name
    author_email="francesco.diotallevi@uni-a.de; irving.reascos.valencia@uni-a.de",  # Author's email
    url="https://github.com/qcode-uni-a/PySW",  # Package URL (if available)
    packages=find_packages(),     # Automatically find packages in your project
    install_requires=[
        "ipython",
        "multimethod",
        "numpy",
        "sympy",
        "tabulate",
        "tqdm",
    ],  # Specify dependencies
    classifiers=[                 # Metadata for the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Example license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',      # Specify minimum Python version
)
