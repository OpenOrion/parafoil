

from setuptools import setup

setup(
   name='parafoil',
   version='2.0',
   description='the open source parametric airfoil generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['parafoil', 'parafoil.airfoils', 'parafoil.passages'],
   install_requires=[
    "numpy",
    "scipy",
    "dacite",
    "ezmesh @ git+https://github.com/Turbodesigner/ezmesh.git",
    "paraflow @ git+https://github.com/Turbodesigner/paraflow.git"
   ]
)