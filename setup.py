from setuptools import setup, find_packages

setup(
    name="PolarCam",
    version="0.8.0",
    author="Daping Xu",
    author_email="daping.xu@physics.ox.ac.uk",
    description="A polarization camera control and analysis tool.",
    packages=find_packages(),
    install_requires=[
        "PySide6",
        "opencv-python",
        "numpy",
        "scikit-image",
        "matplotlib",
        "scipy",
        "ids_peak",
    ],
    entry_points={
        'console_scripts': [
            'polarcam=main:main',
        ],
    },
)
