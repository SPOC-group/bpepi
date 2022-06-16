from distutils.core import setup

setup(
    name="bpepi",
    version="0.1dev",
    packages=["bpepi"],
    install_requires=["numpy", "networkx"],  # external packages as dependencies
    license="Apache License Version 2.0,",
    long_description=open("README.md").read(),
)
