from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="scbsp",
    version='{{VERSION_PLACEHOLDER}}',
    description="A package that efficiently computes p-values for a given set of genes based on input matrices representing cell coordinates and gene expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["scbsp"],
    url="https://github.com/YQ-Wang/scBSP",
    author="Jinpu Li, Yiqing Wang",
    author_email="lijinp@health.missouri.edu, yiqing@wangemail.com",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy >= 1.26.2", "pandas >= 2.1.3", "scipy >= 1.11.3", "scikit-learn >= 1.3.2", "hnswlib >= 0.8.0"],
    extras_require={
        "dev": ["mypy>=1.7.0"],
    },
    python_requires=">=3.6",
)