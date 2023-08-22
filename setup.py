import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msbuddy",
    version="0.1.0",
    author="Shipei Xing",
    author_email="philipxsp@hotmail.com",
    description="Bottom-up MS/MS interrogation-based molecular formula annotation for mass spectrometry data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Philipbear/msbuddy",
    project_urls={
        "Bug Tracker": "https://github.com/Philipbear/msbuddy/issues",
        "Documentation": "https://Philipbear.github.io/msbuddy_Documentation/"
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
    install_requires=[
        "brain-isotopic-distribution",
        "numpy",
        "numba",
        "requests",
        "joblib",
        "gdown"
        "pathlib",
        "chemparse",
        "scipy",
        "scikit-learn",
        "tqdm",
        "timeout-decorator",
    ],
    python_requires=">=3.8"
)
