

from setuptools import setup

setup(
   name='parafoil',
   version='1.0',
   description='the open source parametric airfoil generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['parafoil'],
   install_requires=[
    "numpy",
    "scipy"
   ]
)