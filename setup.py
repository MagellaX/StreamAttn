from setuptools import setup, find_packages

setup(
    name="StreamAttn",
    version="0.1.0",
    description="Multi-GPU FlashAttention with Triton & PyTorch Distributed",
    author="Yash solanki",
    author_email="alphacr792@gmail.com",
    url="https://github.com/yourusername/StreamAttn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "triton>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
