from setuptools import setup, find_packages

setup(
    name="vanilla_gan",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ipykernel",
        "numpy",
        "torch",
        "torchvision",
        "matplotlib",
    ],
)
