# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="amlpds",
    version="1.0.0",
    author="Revan Sai Ande",
    author_email="92206451@iu-study.org",
    description="Adversarial Machine Learning for Phishing Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AMLPDS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "amlpds=src.cli:main",
            "amlpds-train=src.scripts.train_model:main",
            "amlpds-evaluate=src.scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "amlpds": ["templates/*.html", "static/**/*"],
    },
)
