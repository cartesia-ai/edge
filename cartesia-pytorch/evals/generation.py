# Copyright (c) 2024, Aviv Bick, Kevin Li.

import argparse
import time
from functools import partial

import torch
from transformers import AutoTokenizer

from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Rene.rene import ReneLMHeadModel
from cartesia_pytorch.quantize_model import QuantizationConfig, ModelQuantizer, quantize_model_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Rene Descartes was")
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="Llamba-1B",
        choices=["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
    )
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    # Sampling arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    
    # Quantization arguments
    parser.add_argument("--quantize", action="store_true", help="Enable model quantization")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8, 16], help="Quantization bit width")
    parser.add_argument("--group_size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric quantization")
    parser.add_argument("--save_quantized", type=str, default="", help="Path to save the quantized model")
    
    return parser.parse_args()


@torch.inference_mode()
def time_bench(args, input_ids, generate_fn):
    """Benchmark the generation time."""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeats):
        out = generate_fn(input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen)
    torch.cuda.synchronize()

    # Print stats
    print(f"\nTiming results for {args.model} model:")
    print(
        f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"
    )
    print(f"prompt processing + decoding time: {(time.time() - start) / args.repeats * 1000:.0f}ms")


def choose_model(args):
    """Load the model and tokenizer based on the model name."""
    name = args.model
    # Load model
    if name == "Llamba-1B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-1B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    elif name == "Llamba-3B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-3B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    elif name == "Llamba-8B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    elif name == "Rene":
        model = ReneLMHeadModel.from_pretrained("cartesia-ai/Rene-v0.1-1.3b-pytorch")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    else:
        raise NotImplementedError
    
    # Apply quantization if requested
    if args.quantize:
        print(f"Quantizing model to {args.bits} bits...")
        config = QuantizationConfig(
            bits=args.bits,
            group_size=args.group_size,
            sym=args.symmetric,
            per_channel=True,
            use_cuda_kernels=device == "cuda"
        )
        model = ModelQuantizer.quantize_model(model, config)
        
        # Save quantized model if requested
        if args.save_quantized:
            print(f"Saving quantized model to {args.save_quantized}")
            ModelQuantizer.export_model(model, args.save_quantized, {
                'original_model': args.model,
                'quantization_config': {
                    'bits': args.bits,
                    'group_size': args.group_size,
                    'symmetric': args.symmetric
                }
            })

    return model, tokenizer


@torch.inference_mode()
def main():
    """Main function for generation benchmarking."""
    # Parse arguments
    args = parse_args()
    torch.manual_seed(args.seed)

    # Load model
    model, tokenizer = choose_model(args)

    # Prepare model
    model.to(device=device)
    if not args.quantize:  # Only set dtype if not quantized
        model.to(dtype=getattr(torch, args.dtype))
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Tokenize prompt
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (1, args.promptlen), dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)

    # Prepare generation function
    generate_fn = partial(
        model.generate,
        cg=args.model in ["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate
    out = generate_fn(input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen)

    if args.prompt is not None:
        print(
            "Generated text:\n",
            tokenizer.batch_decode(sequences=out.sequences.tolist(), skip_special_tokens=True)[0],
        )

    time_bench(args, input_ids, generate_fn)


if __name__ == "__main__":
    main()
