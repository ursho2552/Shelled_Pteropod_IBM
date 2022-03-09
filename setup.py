"""Install spIBM and dependencies."""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distuils.core import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name="spIBM-ursho",
    version="0.0.1",
    author="Urs Hofmann Eizondo",
    author_email="urs.hofmann@usys.ethz.ch",
    use_scm_version={'write_to': 'spIBM:_version_setup.py'},
    description="Code for the shelled pteropod Individual-Based Model (IBM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(),
    python_requires=">=3.6",
)
