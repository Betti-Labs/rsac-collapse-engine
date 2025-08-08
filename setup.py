#!/usr/bin/env python3
"""
Setup script for RSAC (Recursive Symbolic Attractor Computation)
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rsac-collapse-engine",
    version="1.0.0",
    author="Gregory Betti",
    author_email="gregory@betti-labs.com",
    description="RSAC: Recursive Symbolic Attractor Computation for Boolean Satisfiability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Betti-Labs/rsac-collapse-engine",
    project_urls={
        "Bug Tracker": "https://github.com/Betti-Labs/rsac-collapse-engine/issues",
        "Documentation": "https://github.com/Betti-Labs/rsac-collapse-engine/blob/main/README.md",
        "Source Code": "https://github.com/Betti-Labs/rsac-collapse-engine",
        "Paper": "https://arxiv.org/abs/2025.XXXX",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsac-benchmark=rsac.benchmarks:main",
            "rsac-stress-test=stress_test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rsac": ["*.md", "*.txt"],
    },
    keywords=[
        "sat", "boolean-satisfiability", "constraint-satisfaction", 
        "search-algorithms", "symbolic-computation", "artificial-intelligence"
    ],
    zip_safe=False,
)