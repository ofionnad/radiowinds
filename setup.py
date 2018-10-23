import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="radiowinds",
    version="0.0.2",
    author="Dualta O Fionnagain",
    author_email="ofionnad@tcd.ie",
    description="A package to calculate the thermal free-free emission from stellar winds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dualta93/radiowinds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
