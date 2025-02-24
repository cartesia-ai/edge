# Models

This repository contains the PyTorch implementation of the Rene and Llamba models, which are large-scale language models trained by [Cartesia](https://cartesia.ai).

### Rene

Rene is a 1.3 billion-parameter language model, which is the first model in a series of models trained by [Cartesia](https://cartesia.ai).
Rene has a hybrid architecture based on [Mamba-2](https://arxiv.org/abs/2405.21060), with feedforward and sliding window attention layers interspersed.
It uses the [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) tokenizer.
Rene was pretrained on 1.5 trillion tokens of the [Dolma-1.7](https://huggingface.co/datasets/allenai/dolma) dataset.
For more details, see our [blog post](https://cartesia.ai/blog/on-device).

### Llamba
The Llamba model series is a family of highly efficient recurrent language models distilled from [meta-llama/Llama-3.x](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) into the Mamba-2 architecture, developed in collaboration between Cartesia and CMU’s [Goomba Lab](https://goombalab.github.io). 
The series includes [Llamba-1B](https://huggingface.co/cartesia-ai/Llamba-1B), [Llamba-3B](https://huggingface.co/cartesia-ai/Llamba-3B), and [Llamba-8B](https://huggingface.co/cartesia-ai/Llamba-8B), delivering high inference throughput while maintaining competitive benchmark performance. 
Llamba models scale linearly with input length and were trained on 8B, 10B, and 12B tokens, respectively, demonstrating the effectiveness of [distillation](https://arxiv.org/abs/2408.10189) in large language models.

## Usage
This is the PyTorch version of the package, and it's intended to run on CUDA devices.
For use on Mac computers, please install [the native MLX version](../cartesia-mlx) instead.

### Installation
The Rene model depends on the `cartesia-pytorch` package, which can be installed with `pip` as follows:
```shell
pip install --no-binary :all: cartesia-pytorch
```

### Generation example
```shell
python -m evals.generation \
--model Rene \
--prompt "Rene Descartes was" \
--promptlen 100 \
--genlen 100 \
--dtype bfloat16 \
--temperature 1.0 \
--top_k 1 \
--top_p 0.99 \
--min_p 0.0 \
--repetition_penalty 1.0
```

To generate using another model, simply replace `Rene` with the desired model name, e.g. `Llamba-8B`.
```shell
python -m evals.generation \
--model Llamba-8B \
--prompt "My favorite book is" \
--promptlen 100 \
--genlen 100 \
--dtype bfloat16 \
--temperature 1.0 \
--top_k 1 \
--top_p 0.99 \
--min_p 0.0 \
--repetition_penalty 1.0298
```

### Evaluation example
You can use our `cartesia_lm_eval` wrapper around the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to evaluate our model on standard text benchmarks. Example command (clone this repo and run the below from within the `cartesia-pytorch` directory):
```shell
python -m evals.cartesia_lm_eval --model rene_ssm --model_args pretrained=cartesia-ai/Rene-v0.1-1.3b-pytorch,trust_remote_code=True --trust_remote_code --tasks copa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --cache_requests true --batch_size auto:4 --output_path outputs/rene_evals/
```

```shell
python -m evals.cartesia_lm_eval --model llamba_ssm --model_args pretrained=cartesia-ai/Llamba-8B,trust_remote_code=True --trust_remote_code --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,mmlu --cache_requests true --batch_size auto:4 --output_path outputs/llamba_evals/
```

## About Cartesia
At [Cartesia](https://cartesia.ai/), we're building real-time multimodal intelligence for every device.
