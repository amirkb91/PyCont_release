from setuptools import setup, find_packages

# DEFINE VERSION NUMBER
major = "2"
minor = "0"
patch = "0"

setup(
    name="PyCont_lib_release",
    version=major + "." + minor + "." + patch,
    license="",
    description="python continuation code - Release",
    packages=find_packages(),
    install_requires=[],
)
