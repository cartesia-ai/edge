# Cartesia Metal

This package contains Metal kernels for fast on-device SSM inference on Apple silicon. 

## Installation
To install this package, first install Xcode, which can be downloaded from https://developer.apple.com/xcode/.
Accept the license agreement with:
```shell 
sudo xcodebuild -license
```

We recommend using an environment management tool such as `conda` or `virtualenv`.

Note: This package has been tested on macOS Sonoma 14.1 with the M3 chip.

### Conda installation instructions
Create and activate a new conda environment. Ensure that it is an `arm64` environment. Example:
```shell
CONDA_SUBDIR=osx-arm64 conda create -n cartesia_mlx python=3.11
conda activate cartesia_mlx
```

In your conda environment, install the `nanobind` package (exact version required), followed by `cartesia-metal`:
```shell 
pip install nanobind@git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-metal
```

### Virtualenv installation instructions
Create and activate a new virtualenv:
```shell
python -m venv cartesia_mlx
source cartesia_mlx/bin/activate
```

In your virtual environment, install the `wheel` package, and the `nanobind` package (exact version required), followed by `cartesia-metal`:
```shell
pip install wheel
pip install nanobind@git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-metal --no-build-isolation
```
