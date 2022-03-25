'''
#       Install Project Requirements 
'''
from setuptools import setup, find_packages

setup(name='rsoccer-gym',
    url="https://github.com/robocin/rSoccer",
    description="SSL and VSS robot soccer gym environments",
    packages=[package for package in find_packages() if package.startswith("rsoccer_gym")],
    install_requires=['gym==0.21.0', 'rc-robosim>=1.2.0', 'pyglet', 'protobuf']
)
