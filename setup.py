import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkg_location = 'src'
pkg_name = 'fincnn'

setuptools.setup(
    name=pkg_name,
    version="1.9.1",
    description="Technical analysis CNN models for stock return prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anakazh/tech_analysis_cnn",
    py_modules=[pkg_name],
    package_dir={'': pkg_location},
    packages=setuptools.find_packages(where=pkg_location),
    install_requires=[
        'pandas>=1.3.4',
        'numpy>=1.21.4',
        'tqdm>=4.62.3',
        'mplfinance>=0.12.7a17',
        'tensorflow>=2.7.0',
        'sklearn',
        'SQLAlchemy>=1.4.27',
        'matplotlib>=3.4.3',
        'argparse'
                    ],
    python_requires=">=3.7",
)
