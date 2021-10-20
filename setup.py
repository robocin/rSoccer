'''
#       Install Project Requirements 
'''
from setuptools import setup, find_packages

setup(name='rsoccer-gym',
    version='0.1.0a4',
    url="https://github.com/robocin/rSoccer",
    description="SSL and VSS robot soccer gym environments",
    packages=[package for package in find_packages() if package.startswith("rsoccer_gym")],
    install_requires=['gym', 'rc-robosim==0.0.8a0', 'pyglet', 'protobuf']
)
