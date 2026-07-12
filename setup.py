"""
setup.py — enables `pip install -e .` for local development
and `pip install .` for packaging in Week 9.
"""
from setuptools import setup, find_packages

setup(
    name="agentlens",
    version="1.0.0",
    description="Multi-Agent Failure Attribution, Trace Diffing & Explainability Platform",
    author="Anks",
    python_requires=">=3.12",
    packages=find_packages(exclude=["tests*", "sample_data*", "docs*"]),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-groq>=0.2.0",
        "pydantic>=2.7.0",
        "sentence-transformers>=3.0.0",
        "streamlit>=1.36.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
    ],
    entry_points={
        "console_scripts": [
            "agentlens=app.main:main",
        ],
    },
)
