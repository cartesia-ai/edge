#!/usr/bin/env python3
"""
Token ID Dumper for Rene Model

This script tokenizes text using the OLMo tokenizer (same as Rene model)
and outputs token IDs to a binary file for use with the OpenCL C++ driver.
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
    import os
    from huggingface_hub import login
except ImportError:
    print("Error: transformers library not found. Install with: pip install transformers")
    sys.exit(1)


def tokenize_text(text: str, tokenizer_name: str = "allenai/OLMo-1B-hf", token: str = None):
    """Tokenize text and return list of token IDs and tokenizer."""
    print(f"Loading tokenizer: {tokenizer_name}")
    
    # Try to get token from environment or use provided token
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Try local files first (faster, works if already downloaded)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=token,
            local_files_only=True
        )
        print("Loaded tokenizer from local cache")
    except Exception as e:
        # If local files don't work, try downloading (requires auth if gated)
        if "404" in str(e) or "not found" in str(e).lower():
            print("Tokenizer not in cache, attempting to download...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    token=token,
                    local_files_only=False
                )
            except Exception as e2:
                if "401" in str(e2) or "Unauthorized" in str(e2):
                    print("\nError: Authentication required to access this tokenizer.")
                    print("Please authenticate with Hugging Face:")
                    print("  1. Get a token from: https://huggingface.co/settings/tokens")
                    print("  2. Run: huggingface-cli login")
                    print("  3. Or set environment variable: export HF_TOKEN=your_token_here")
                    print("\nAlternatively, if you've already accepted the model terms on Hugging Face,")
                    print("make sure you're logged in via: huggingface-cli login")
                raise e2
        else:
            raise
    
    print(f"Tokenizing text: '{text}'")
    result = tokenizer(text)
    # input_ids is already a list when tokenizing a single string
    tokens = result.input_ids if isinstance(result.input_ids, list) else result.input_ids[0]
    
    print(f"Generated {len(tokens)} token IDs")
    print(f"Token IDs: {tokens[:20]}..." if len(tokens) > 20 else f"Token IDs: {tokens}")
    
    return tokens, tokenizer


def write_tokens_binary(tokens: list, output_path: Path):
    """Write token IDs as binary file (little-endian int32)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Write number of tokens as int32
        f.write(struct.pack('<i', len(tokens)))
        # Write each token ID as int32
        for token_id in tokens:
            f.write(struct.pack('<i', token_id))
    
    print(f"Wrote {len(tokens)} token IDs to {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


def write_tokens_text(tokens: list, output_path: Path):
    """Write token IDs as text file (one per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"{len(tokens)}\n")  # First line: count
        for token_id in tokens:
            f.write(f"{token_id}\n")
    
    print(f"Wrote {len(tokens)} token IDs to {output_path}")


def write_tokens_readable(tokens: list, tokenizer, output_path: Path):
    """Write token IDs with their text representations in human-readable format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Token Count: {len(tokens)}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Index':<8} {'Token ID':<12} {'Token Text':<40}\n")
        f.write(f"{'-'*60}\n")
        
        for idx, token_id in enumerate(tokens):
            # Decode the token ID to get the actual token text
            token_text = tokenizer.decode([token_id])
            # Clean up the token text for display
            # Replace newlines and other control characters
            token_text_clean = repr(token_text) if any(ord(c) < 32 or c in ['\n', '\r', '\t'] for c in token_text) else token_text
            f.write(f"{idx:<8} {token_id:<12} {token_text_clean}\n")
        
        f.write(f"{'='*60}\n")
        f.write(f"\nFull tokenized sequence:\n")
        f.write(f"Token IDs: {tokens}\n")
        full_text = tokenizer.decode(tokens)
        f.write(f"Decoded text: {repr(full_text)}\n")
    
    print(f"Wrote readable token mapping to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text for Rene model and dump token IDs to file"
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text prompt to tokenize"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("prompt_tokens.bin"),
        help="Output file path (default: prompt_tokens.bin)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-1B-hf",
        help="Tokenizer name (default: allenai/OLMo-1B-hf)"
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Also write text format to this file (optional)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for authentication (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Tokenize
    tokens, tokenizer = tokenize_text(args.text, args.tokenizer, token=args.token)
    
    # Write binary format
    write_tokens_binary(tokens, args.output)
    
    # Write human-readable mapping file (prompt_tokens.txt)
    # Derive the .txt filename from the binary output filename
    readable_output = args.output.with_suffix('.txt')
    write_tokens_readable(tokens, tokenizer, readable_output)
    
    # Write text format if requested (simple list format)
    if args.text_output:
        write_tokens_text(tokens, args.text_output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

