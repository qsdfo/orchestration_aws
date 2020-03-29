from setuptools import setup

setup(
    name='DCPC',  # This is the name of your PyPI-package.
    version='0.1',
    url='git@github.com:SonyCSLParis/DCPC.git',
    author='Gaetan Hadjeres & Léopold Crestel',
    author_email='gaetan.hadjeres@sony.com',
    license='BSD',
    packages=['DCPC'], install_requires=['pymongo', 'music21', 'sshtunnel', 'numpy',
                                                   'torch', 'tqdm']
)