

from setuptools import setup

setup(
   name='parafoil',
   version='2.5',
   description='the open source parametric airfoil generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['parafoil', 'parafoil.airfoils', 'parafoil.passages'],
   install_requires=[
    "numpy",
    "scipy",
    "dacite",
    "ezmesh @ git+https://github.com/OpenOrion/ezmesh.git",
    "paraflow @ git+https://github.com/OpenOrion/paraflow.git@2.0.0"
   ]
)
