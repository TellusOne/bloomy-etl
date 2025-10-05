from setuptools import setup, find_packages

setup(
    name="bloomy",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "earthaccess",
        "xarray",
        "rasterio",
        "shapely",
        "pyproj",
        "numpy",
        "tqdm",
        "aiohttp",
        "python-dotenv"
    ]
)