from setuptools import setup

setup(name='gym_vss',
      version='0.0.1',
      install_requires=['gym', 'gym[atari]',
                        'opencv-python', 'protobuf',
                        'pyzmq', 'torch', 'torchvision',
                        'joblib', 'sslclient']
      )
