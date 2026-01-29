"""
MCM/ICM Automation System Setup
"""

from setuptools import setup, find_packages

setup(
    name='mcm-icm-skills',
    version='2.0.0',
    description='MCM/ICM Paper Writing Automation System',
    author='MCM Automation Team',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'pandas>=2.2.0',
        'scipy>=1.12.0',
        'sympy>=1.12',
        'scikit-learn>=1.4.0',
        'matplotlib>=3.8.0',
        'seaborn>=0.13.0',
        'pyyaml>=6.0.1',
        'requests>=2.31.0',
        'aiohttp>=3.9.0',
        'pydantic>=2.5.0',
    ],
    extras_require={
        'optimization': [
            'cvxpy>=1.4.0',
            'pulp>=2.7.0',
            'ortools>=9.8',
        ],
        'ml': [
            'xgboost>=2.0.0',
            'lightgbm>=4.0.0',
            'prophet>=1.1.5',
            'darts>=0.28.0',
        ],
        'network': [
            'networkx>=3.2',
            'python-igraph>=0.11.0',
        ],
        'sensitivity': [
            'SALib>=1.5.0',
        ],
        'nlp': [
            'language-tool-python>=2.8.0',
            'spacy>=3.7.0',
        ],
        'document': [
            'PyPDF2>=3.0.0',
            'pdfplumber>=0.10.0',
            'pylatex>=1.4.2',
            'bibtexparser>=2.0.0',
        ],
        'dev': [
            'pytest>=8.0.0',
            'pytest-asyncio>=0.23.0',
            'black>=24.0.0',
            'mypy>=1.8.0',
        ],
        'all': [
            'cvxpy>=1.4.0',
            'pulp>=2.7.0',
            'ortools>=9.8',
            'xgboost>=2.0.0',
            'prophet>=1.1.5',
            'networkx>=3.2',
            'SALib>=1.5.0',
            'language-tool-python>=2.8.0',
            'pdfplumber>=0.10.0',
            'pylatex>=1.4.2',
        ],
    },
    entry_points={
        'console_scripts': [
            'mcm-run=scripts.test_runner:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
