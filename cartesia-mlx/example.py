import argparse

import mlx.core as mx

import cartesia_mlx as cmx

parser = argparse.ArgumentParser(
    description="Simple example script to run a Cartesia MLX language model."
)
parser.add_argument(
    "--model", default="cartesia-ai/Rene-v0.1-1.3b-4bit-mlx", help="The model name."
)
parser.add_argument("--prompt", default="Rene Descartes was")
parser.add_argument(
    "--max-tokens", type=int, default=500, help="Maximum number of tokens to generate."
)
parser.add_argument(
    "--top-p",
    type=float,
    default=0.99,
    help="Top-p sampling (a value of 1 is equivalent to standard sampling).",
)
parser.add_argument(
    "--temperature", type=float, default=0.85, help="Temperature scaling parameter."
)
args = parser.parse_args()

model = cmx.from_pretrained(args.model)
model.set_dtype(mx.float32)

prompt = args.prompt

print(prompt, end="", flush=True)
for text in model.generate(
    prompt,
    max_tokens=args.max_tokens,
    eval_every_n=5,
    verbose=True,
    top_p=args.top_p,
    temperature=args.temperature,
):
    print(text, end="", flush=True)
