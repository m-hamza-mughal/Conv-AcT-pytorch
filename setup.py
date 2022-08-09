from setuptools import find_packages, setup


setup(
    name='conv_act',
    packages=find_packages(exclude="configs"),
    version='0.1.0',
    description='Human Activity Recognition (HAR) with Transformer based on Convolutional Features.',
    author='Hamza Mughal',
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "pyyaml",
        "av",
        "fvcore"
    ],
    license='MIT',
)