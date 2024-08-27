# Cartesia Metal

This package contains Metal kernels for fast on-device SSM inference on Apple silicon. 

## Installation
To install this package, first install Xcode, which can be downloaded from https://developer.apple.com/xcode/.
Accept the license agreement with:
```shell 
sudo xcodebuild -license
```

Install the dependency `nanobind` (exact version required), followed by `cartesia-metal`:
```shell 
pip install nanobind@git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-metal
```

Note: This package has been tested on macOS Sonoma 14.1 with the M3 chip.
