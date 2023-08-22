import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msbuddy",
    version="0.0.1",
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
    install_requires=[
        "numpy",
        "joblib",
        "gdown"
        "pathlib",
        "chemparse",
        "scipy",
        "scikit-learn",
        "tqdm"
    ],
    python_requires=">=3.6",
    include_package_data=True
)