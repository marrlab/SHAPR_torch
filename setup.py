import os
from setuptools import setup, find_packages
from pathlib import Path

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'shapr_torch',
    version='0.0.1',
    author='Dominik Waibel, Bastian Rieck',
    author_email='dominik.waibel@helmholtz-muenchen.de',
    license='MIT',
    keywords='Computational Biology Deep Learning',
    packages=find_packages(exclude=['doc*', 'test*']),
    url='https://github.com/marrlab/SHAPR_torch/',
    install_requires=['torch', 'imageio', 'scikit-image', 'scikit-learn', 'torch-topological', 'tensorflow', 'pytorch_lightning', 'torchvision'],
    classifiers=[
	'Development Status :: 3 - Alpha',
	'Topic :: Scientific/Engineering :: Image Recognition',
	'License :: OSI Approved :: MIT License',
	]
)
