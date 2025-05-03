from setuptools import setup, find_packages

setup(
    name="luminara",
    version="0.1.0",
    description="Conditional Random Field framework for AI investment worthiness prediction",
    author="Lianghao Chen",
    author_email="rickchen19970829@outlook.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pgmpy>=0.1.20",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "networkx>=2.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
