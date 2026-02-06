from setuptools import setup, find_packages

setup(
    name="shared_files",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",                      # optional dependencies
        "gymnasium",
        "safety_gymnasium",
        "numpy",
        "omnisafe",
        "mujoco",
    ],
)