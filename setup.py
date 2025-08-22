"""
Setup script for StreamAttention
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

# Separate core and optional requirements
core_requirements = []
optional_requirements = {
    "transformers": [],
    "distributed": [],
    "dev": [],
    "docs": [],
}

current_section = "core"
for req in requirements:
    if "# Optional:" in req:
        if "HuggingFace" in req:
            current_section = "transformers"
        elif "distributed" in req:
            current_section = "distributed"
    elif "# Development" in req:
        current_section = "dev"
    elif "# Documentation" in req:
        current_section = "docs"
    elif req and not req.startswith("#"):
        if current_section == "core":
            core_requirements.append(req)
        else:
            optional_requirements[current_section].append(req)

# Combine all optional requirements
all_optional = []
for deps in optional_requirements.values():
    all_optional.extend(deps)

optional_requirements["all"] = all_optional

setup(
    name="stream-attention",
    version="1.0.0",
    author="StreamAttention Team",
    author_email="streamattention@example.com",
    description="Production-ready multi-GPU FlashAttention implementation with support for extremely long contexts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/StreamAttention",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/StreamAttention/issues",
        "Documentation": "https://streamattention.readthedocs.io",
        "Source Code": "https://github.com/yourusername/StreamAttention",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        **optional_requirements,
        "hf": ["transformers>=4.39"],
    },
    include_package_data=True,
    package_data={
        "stream_attention": ["*.yaml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "stream-attention-benchmark=stream_attention.benchmarks.benchmark_suite:main",
            "stream-attention-test=stream_attention.benchmarks.accuracy_test:main",
        ],
    },
    keywords=[
        "attention",
        "transformer",
        "flash-attention",
        "deep-learning",
        "pytorch",
        "triton",
        "gpu",
        "long-context",
        "llm",
    ],
)
