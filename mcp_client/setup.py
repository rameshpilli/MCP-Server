from setuptools import setup, find_packages

setup(
    name="mcp-client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.27.0",
        "pydantic>=2.6.0",
        "python-dotenv>=1.0.0",
        "typer>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "mcp=mcp_client.cli:app",
        ],
    },
    author="Your Name",
    author_email="your.email@company.com",
    description="Client SDK for Model Control Platform (MCP)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rameshpilli/MCP-Server",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 