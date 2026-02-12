from setuptools import setup, find_packages
with open("README.md", "r") as fh: 
    long_description = fh.read() 
setup(
    name = 'sona-oversampling',
    version = "0.0.1",
    description = "Stochastic directional Oversampling using Negative Anomalous scores (SONA)",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires = [
        'numpy>=1.23','scipy>=1.8.0',
    ],
)