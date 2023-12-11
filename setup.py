from setuptools import setup, find_packages

# DEFINE VERSION NUMBER
major = "1"
minor = "0"
patch = "0"

setup(
    name="PyCont_Release",
    version=major + "." + minor + "." + patch,
    license="",
    description="core python continuation code - Release",
    packages=find_packages(),
    install_requires=[],
)
