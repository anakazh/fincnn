import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tech-analysis-cnn",
    version="0.2",
    description="Technical analysis CNN models for stock return prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anakazh/tech_analysis_cnn",
    packages=setuptools.find_packages(include=['tech_analysis_cnn']),
    install_requires=[
        'pandas>=1.3.4',
        'numpy>=1.21.4',
        'tqdm>=4.62.3',
        'mplfinance>=0.12.7a17',
        'tensorflow>=2.7.0',
        'sklearn',
        'SQLAlchemy>=1.4.27',
        'matplotlib>=3.4.3'
                    ],
    python_requires=">=3.7",
)
