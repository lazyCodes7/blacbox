from setuptools import find_packages, setup
from os import path


with open('requirements.txt') as f:
    reqs = f.read().splitlines()
with open('README.md', encoding='utf-8') as fr:
    long_description = fr.read()

setup(
    name="blacbox",
    version="0.1.0",
    description="A visualization library to make CNNs more interpretable",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lazyCodes7/blacbox",
    author="Rishab Mudliar",
    author_email="rishabmudliar@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
)