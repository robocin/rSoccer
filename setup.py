"""
#       Install Project Requirements 
"""
from setuptools import setup, find_packages

setup(
    name="rsoccer-gym",
    url="https://github.com/robocin/rSoccer",
    description="SSL and VSS robot soccer gym environments",
    packages=[
        package for package in find_packages() if package.startswith("rsoccer_gym")
    ],
    install_requires=[
        "gymnasium >= 0.28.1",
        "rc-robosim >= 1.2.0",
        "pygame >= 2.1.3",
        "protobuf == 3.20",
    ],
)
