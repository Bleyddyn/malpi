import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="malpi",
    version="0.0.1",
    author="Andrew Salamon",
    author_email="bleyddyn.aprhyx@gmail.com",
    description="A library for use with the DonkeyCar project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bleyddyn/malpi",
    packages=["malpi"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7.10',
)
