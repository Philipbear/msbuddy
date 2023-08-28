import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msbuddy",
    version="0.1.1",
    author="Shipei Xing",
    author_email="philipxsp@hotmail.com",
    description="Bottom-up MS/MS interrogation-based molecular formula annotation for mass spectrometry data.",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["msbuddy"],
    include_package_data=True,
    package_data={
        'msbuddy': ['data/*'],  # Include all files in msbuddy/data
    },
    install_requires=[
        "brain-isotopic-distribution",
        "numpy",
        "numba",
        "requests",
        "joblib",
        "gdown",
        "chemparse",
        "scipy",
        # "scikit-learn",
        "tqdm",
        "timeout_decorator",
        "pandas",
    ],
    python_requires=">=3.8"
)
