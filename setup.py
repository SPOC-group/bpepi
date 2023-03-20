from distutils.core import setup

setup(
    name="bpepi",
    version="0.2dev",
    packages=["bpepi", "bpepi.Modules"],
    install_requires=[
        "numpy",
        "networkx",
        "torch",
    ],  # external packages as dependencies
    license="Apache License Version 2.0,",
    long_description=open("README.md").read(),
)
