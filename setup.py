from setuptools import setup, dist , find_packages

try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(["cython>=0.28.0"])
    from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
try:
    from numpy import get_include as np_get_include
except ImportError:
    dist.Distribution().fetch_build_eggs(["numpy"])
    from numpy import get_include as np_get_include

import causalforge

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

with open("requirements-test.txt") as f:
    requirements_test = f.readlines()

packages = find_packages(exclude=["tests", "tests.*"])

setup(
    name="causalforge",
    version=causalforge.__version__,
    author="Gino Tesei, Jey Kottalam",
    author_email="",
    description="Python Package for Causal Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthem-ai/causalflow",
    packages=packages,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    setup_requires=[
        "setuptools",
        "cython",
        "numpy",
        "scikit-learn",
        "tensorflow", 
        "keras", 
        "torch"
    ],
    install_requires=requirements,
    tests_require=requirements_test,
    license="MIT",
    include_package_data=True,
    include_dirs=[np_get_include()] 
)
