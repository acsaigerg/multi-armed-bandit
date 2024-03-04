from setuptools import setup, find_packages

setup(
    name="multi_armed_bandit",
    version="0.0.1",
    description="A simple gym implementation of multi-armed bandit.",
    author="Gergely Acsai",
    packages=find_packages(),
    install_requires=["gymnasium~=0.28.1", "matplotlib~=3.7.2", "numpy~=1.25.0"],
)
