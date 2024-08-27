---
license: apache-2.0
language:
- en
datasets:
- allenai/dolma
tags:
- rene
- mamba
- cartesia
---

# Model Card for Rene

Rene is a 1.3 billion-parameter language model trained by [Cartesia](https://cartesia.ai).
Rene has a hybrid architecture based on [Mamba-2](https://arxiv.org/abs/2405.21060), with feedforward and sliding window attention layers interspersed.
It uses the [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) tokenizer.
Rene was pretrained on 1.5 trillion tokens of the [Dolma-1.7](https://huggingface.co/datasets/allenai/dolma) dataset.
For more details, see our [blog post](https://cartesia.ai/blog/on-device).

## Usage
### Installation
The Rene model depends on the `cartesia-pytorch` package, which can be installed with `pip` as follows:
```shell
pip install --no-binary :all: cartesia-pytorch
```

### Generation example
```python
from cartesia_pytorch import ReneLMHeadModel
from transformers import AutoTokenizer

model = ReneLMHeadModel.from_pretrained("cartesia-ai/Rene-v0.1-1.3b-pytorch").half().cuda()
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
in_message = ["Rene Descartes was"]
inputs = tokenizer(in_message, return_tensors="pt")
outputs = model.generate(inputs.input_ids.cuda(), max_length=50, top_k=100, top_p=0.99)
out_message = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(out_message)
# Example output: "Rene Descartes was a French mathematician, philosopher, and scientist. Descartes is famously credited for creating the Cartesian coordinate system: a 3 dimensional representation of points, vectors, and directions. This work is, for the most part" ...
```

### Evaluation example
You can use our `cartesia_lm_eval` wrapper around the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to evaluate our model on standard text benchmarks. Example command (clone this repo and run the below from within the `cartesia-pytorch` directory):
```shell
python -m evals.cartesia_lm_eval --model rene_ssm --model_args pretrained=cartesia-ai/Rene-v0.1-1.3b-pytorch,trust_remote_code=True --trust_remote_code --tasks copa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa --cache_requests true --batch_size auto:4 --output_path outputs/rene_evals/
```
## Results on common benchmarks
| Model                                          | Params (B) | Train Tokens | COPA | HellaSwag | MMLU (5-shot) | PIQA | ARC-e | ARC-c | WinoGrande | OpenBookQA | Average |
|------------------------------------------------|------------|--------------|------|-----------|---------------|------|-------|-------|------------|------------|---------|
| allenai/OLMo-1B-hf                             | 1.2        | 3.0          | 82.0 | 62.9      | 26.2          | 75.1 | 57.4  | 31.1  | 60.0       | 36.2       | 53.9    |
| apple/OpenELM-1\_1B                            | 1.1        | 1.5          | 81.0 | 64.8      | 27.1          | 75.6 | 55.4  | 32.3  | 61.9       | 36.2       | 54.3    |
| state-spaces/mamba2-1.3b                       | 1.3        | 0.3          | 82.0 | 60.0      | 25.8          | 73.7 | 64.2  | 33.3  | 61.0       | 37.8       | 54.7    |
| microsoft/phi-1\_5                             | 1.4        | 0.15         | 79.0 | 62.6      | 42.5          | 75.5 | 73.2  | 48.0  | 72.8       | 48.0       | 62.7    |
| Qwen/Qwen2-1.5B                                | 1.5        | 7.0          | 80.0 | 65.4      | 56.0          | 75.5 | 60.4  | 35.0  | 65.8       | 36.4       | 59.3    |
| RWKV/rwkv-6-world-1b6                          | 1.6        | 1.1          | 84.0 | 58.3      | 25.9          | 73.5 | 56.7  | 34.1  | 60.0       | 37.4       | 53.7    |
| stabilityai/stablelm-2-1\_6b                   | 1.6        | 4.0          | 86.0 | 69.0      | 38.1          | 76.7 | 68.1  | 38.9  | 63.6       | 38.8       | 59.9    |
| HuggingFaceTB/SmolLM-1.7B                      | 1.7        | 1.0          | 76.0 | 65.8      | 29.9          | 76.1 | 73.5  | 46.4  | 60.9       | 42.0       | 58.8    |
| h2oai/h2o-danube2-1.8b-base                    | 1.8        | 3.0          | 82.0 | 72.4      | 39.9          | 77.3 | 69.0  | 39.9  | 63.9       | 41.4       | 60.7    |
| google/recurrentgemma-2b                       | 2.7        | 2.0          | 62.0 | 61.8      | 32.3          | 68.8 | 46.4  | 29.9  | 57.1       | 29.0       | 48.4    |
| cognitivecomputations/TinyDolphin-2.8.1-1.1b   | 1.1        |              | 71.0 | 59.9      | 25.7          | 73.1 | 55.8  | 33.0  | 59.7       | 36.6       | 51.9    |
| cartesia-ai/Rene-v0.1-1.3b-pytorch (OUR MODEL) | 1.3        | 1.5          | 82.0 | 69.4      | 32.6          | 77.5 | 61.7  | 34.4  | 62.9       | 39.2       | 57.5    |

## Bias, Risks, and Limitations
Rene is a pretrained base model which has not undergone any alignment or instruction tuning, and therefore does not have any moderation or safety guarantees. Users should implement appropriate guardrails and moderation mechanisms based on their particular needs in order to ensure responsible and ethical usage.

## About Cartesia
At [Cartesia](https://cartesia.ai/), we're building real-time multimodal intelligence for every device.
