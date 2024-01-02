from __future__ import annotations

from setuptools import find_packages, setup

__version__: str = "0.1.0"


setup(
    name="distorted",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="amirhajibabaei@gmail.com",
    url="https://github.com/amirhajibabaei/Distorted",
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Typing :: Typed",
    ],
    python_requires=">=3.7",
    install_requires=[],
    packages=find_packages(exclude=[]),
)
