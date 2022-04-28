import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from malpi import __version__

setuptools.setup(
    name="malpi",
    version=__version__,
    author="Andrew Salamon",
    author_email="bleyddyn.aprhys@gmail.com",
    description="A library for use with the DonkeyCar project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bleyddyn/malpi",
    packages=["malpi", "malpi.train", "malpi.dkwm", "malpi.ui"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7.10',
)
