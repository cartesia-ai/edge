import os
import subprocess
import sys

from setuptools import setup

NAME = "cartesia-metal"

here = os.path.abspath(os.path.dirname(__file__))

def parse_version(filename):
    about = {}
    with open(filename, "r") as f:
        exec(f.read(), about)
    return about["__version__"]


version = parse_version(os.path.join(here, NAME.replace("-", "_"), "version.py"))

from mlx import extension

if __name__ == "__main__":
    setup(
        name=NAME,
        version=version,
        description="Cartesia MLX Extensions",
        ext_modules=[extension.CMakeExtension("cartesia_metal._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["cartesia_metal"],
        package_data={"cartesia_metal": ["*.so", "*.dylib", "*.metallib"]},
        extras_require={"dev": []},
        zip_safe=False,
        python_requires=">=3.9",
    )
