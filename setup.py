from setuptools import find_packages, setup

setup(
    name="tabular-ml-pipeline",
    version="1.0.0",
    author="Pradeep Kumar Kommuri",
    description="Production-style tabular ML pipeline with PyTorch TabNet",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "xgboost>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black", "ruff"],
    },
)
