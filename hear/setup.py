#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("readme.md", "r").read()

setup(
    name="hear_mst",
    description="Barlow Twins for Audio with Masked Spectrogram Transformer",
    author="Jonah Anton",
    url="https://github.com/jonahanton/SSL_audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/jonahanton/SSL_audio/issues",
        "Source Code": "https://github.com/jonahanton/SSL_audio",
    },
    packages=find_packages(exclude=("tests",)),
    install_requires=["timm", "torchaudio", "einops"]
)