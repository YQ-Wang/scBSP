from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="scbsp",
    version="{{VERSION_PLACEHOLDER}}",
    description="A package that efficiently computes p-values for a given set of genes based on input matrices representing cell coordinates and gene expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["scbsp"],
    url="https://github.com/YQ-Wang/scBSP",
    author="Yiqing Wang, Jinpu Li",
    author_email="yqw@wangemail.com, lijinp@health.missouri.edu",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy >= 1.24.4", "pandas >= 1.3.5", "scipy >= 1.10.1", "scikit-learn >= 1.3.2"],
    extras_require={
        "gpu": ["torch >= 1.10.0"],
    },
    python_requires=">=3.8",
)