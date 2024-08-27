# Cartesia MLX

This package contains implementations for fast on-device SSM inference on Apple silicon. 

## Installation
To install this package, first install Xcode, which can be downloaded from https://developer.apple.com/xcode/.
Accept the license agreement with:
```shell 
sudo xcodebuild -license
```

Install the required dependencies: the exact version of `nanobind`, followed by `cartesia-metal`, and finally `cartesia-mlx`, with the following commands:
```shell 
pip install nanobind@git+https://github.com/wjakob/nanobind.git@2f04eac452a6d9142dedb957701bdb20125561e4
pip install git+https://github.com/cartesia-ai/edge.git#subdirectory=cartesia-metal
pip install cartesia-mlx
```

Note: This package has been tested on macOS Sonoma 14.1 with the M3 chip.

## Models

### Language Models
- `cartesia-ai/Rene-v0.1-1.3b-4bit-mlx` 
- `cartesia-ai/mamba2-130m-8bit-mlx` 
- `cartesia-ai/mamba2-130m-mlx` 
- `cartesia-ai/mamba2-370m-8bit-mlx` 
- `cartesia-ai/mamba2-780m-8bit-mlx` 
- `cartesia-ai/mamba2-1.3b-4bit-mlx` 
- `cartesia-ai/mamba2-2.7b-4bit-mlx` 

## Usage
A simple example script for generation can be found in `cartesia-mlx/example.py`.
Usage example (clone this repo and run the below from within the `cartesia-mlx` directory):
```shell
python example.py --model cartesia-ai/Rene-v0.1-1.3b-4bit-mlx --prompt "Rene Descartes was"
```

You can pass any of the models listed above to the `--model` argument; for a full list of command-line options, pass `--help`.

## Rene in MLX
![Language Model](assets/lm-demo.gif)
