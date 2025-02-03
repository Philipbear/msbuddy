import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msbuddy",
    version="0.3.10",
    author="Shipei Xing",
    author_email="philipxsp@hotmail.com",
    description="Molecular formula annotation for MS-based small molecule analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Philipbear/msbuddy",
    project_urls={
        "Bug Tracker": "https://github.com/Philipbear/msbuddy/issues",
        "Documentation": "https://msbuddy.readthedocs.io/en/latest/"
    },
    scripts=['msbuddy/main_cmd.py'],
    entry_points={
        'console_scripts': [
            'msbuddy=msbuddy.main_cmd:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=["msbuddy"],
    install_requires=[
        "brain-isotopic-distribution",
        "numpy",
        "numba",
        "requests",
        "joblib",
        "gdown",
        "chemparse",
        "scipy",
        "tqdm",
        "pandas"
    ],
    python_requires=">=3.8"
)
