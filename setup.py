from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coder-agent",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A minimalistic coding agent using OpenRouter API and Cerebras provider",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coder-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.31.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "coder=coder.cli:app",
        ],
    },
) 