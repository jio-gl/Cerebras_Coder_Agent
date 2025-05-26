from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coder",
    version="1.0.0",
    author="Jose Ignacio",
    author_email="joseignacio@example.com",
    description="A powerful coding agent with enhanced features for parallel processing, caching, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joseignacio/Cerebras_Hackaton_Coding_Agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "diskcache>=5.6.0",
        "tenacity>=8.1.0",
        "filetype>=1.2.0",
        "pydantic>=1.10.0",
        "pandas>=1.5.0",
        "typer>=0.10.0",
        "rich>=13.3.1",
        "aiofiles>=23.1.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.75.3",
        "click>=8.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "coder=coder.cli:cli",
        ],
    },
)
