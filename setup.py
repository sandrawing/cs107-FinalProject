import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodiff-AsiaUnionCS107",
    version="0.0.15",
    author="AsiaUnionCS107",
    description="An Automatic Differentiation Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AsiaUnionCS107/cs107-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
