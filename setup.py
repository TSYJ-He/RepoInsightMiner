from setuptools import setup, find_packages

setup(
    name='repoinsightminer',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'networkx>=3.3',
        'torch>=2.3.1',
        'torch-geometric>=2.5.3',
        'spacy>=3.7.5',
        'pygithub>=2.3.0',
        'streamlit>=1.37.1',
        'fastapi>=0.112.0',
        'uvicorn>=0.30.3',
        'vaderSentiment>=3.3.2',
        'pylint>=3.2.6',
    ],
    entry_points={
        'console_scripts': [
            'repoinsightminer-cli = cli.main:main',
        ],
    },
    include_package_data=True,
    python_requires='>=3.12',
    description='AI-powered GitHub repo analyzer for insights on bugs, architecture, and contributors.',
    author='Your Name',
    license='MIT',
)