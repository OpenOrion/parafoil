

from setuptools import setup

setup(
   name='parafoil',
   version='3.5.0',
   description='the open source parametric airfoil generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['parafoil', 'parafoil.airfoils', 'parafoil.passages'],
   install_requires=[
    "numpy",
    "scipy",
    "dacite",
    "paraflow @ git+https://github.com/OpenOrion/paraflow.git@3.5.0"
   ]
)
