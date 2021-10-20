'''
#       Install Project Requirements 
'''
from setuptools import setup, find_packages

setup(name='rsoccer_gym',
    version='0.1',
    packages=[package for package in find_packages() if package.startswith("rsoccer_gym")],
    install_requires=['gym', 'rc-robosim', 'pyglet', 'protobuf']
)
