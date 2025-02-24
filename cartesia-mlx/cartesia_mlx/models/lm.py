import time

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.layers.embedding import Embedding
from cartesia_mlx.layers.ffn import SwiGLU
from cartesia_mlx.layers.linear import Linear
from cartesia_mlx.layers.residual_block import ResidualBlock
from cartesia_mlx.layers.sa import SelfAttention
from cartesia_mlx.layers.sequence_model import SequenceModel
from cartesia_mlx.layers.ssd.ssd import SSD
from cartesia_mlx.utils.configure import Inherit, set_cfg, sub_cfg
from cartesia_mlx.utils.sample_utils import categorical_sampling, min_p_sampling, top_p_sampling
from cartesia_mlx.utils.tokenizer import Tokenizer


class LM(nn.Module):
    """Generic language model backbone.

    Example:
        import mlx.core as mx
        import cartesia_mlx as cmx

        model = cmx.from_pretrained('cartesia-ai/mamba2-130m-8bit-mlx')
        model.set_dtype(mx.float32)

        prompt = "Rene Descartes was"

        print(prompt, end="", flush=True)
        for text in model.generate(
            prompt,
            max_tokens=500,
            eval_every_n=5,
            verbose=True,
            top_p=0.99,
            temperature=0.85,
        ):
            print(text, end="", flush=True)
    """

    base_cfg = dict(
        _class_="models.lm.LM",
        quantization_kwargs=None,
        d_model=2048,
        n_tokens=50288,
        eos=50279,
        tokenizer=sub_cfg(Tokenizer.base_cfg),
        embedding=sub_cfg(Embedding.base_cfg, n_tokens=Inherit(), d_model=Inherit()),
        model=sub_cfg(
            SequenceModel.base_cfg,
            n_layer_repeats=4,
            unique_layers=[
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SwiGLU.base_cfg, stateful=False),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SwiGLU.base_cfg, stateful=False),
                sub_cfg(ResidualBlock.base_cfg, layer=SelfAttention.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SwiGLU.base_cfg, stateful=False),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
                sub_cfg(ResidualBlock.base_cfg, layer=SwiGLU.base_cfg, stateful=False),
            ],
        ),
        head=sub_cfg(
            Linear.base_cfg,
            output_dim=Inherit(from_key="n_tokens"),
            input_dim=Inherit(from_key="d_model"),
        ),
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent, instantiate_children=True)

    def generate(
        self,
        prompt,
        max_tokens=500,
        eval_every_n=1,
        verbose=False,
        **sampling_kwargs,
    ):
        """Generates text given an initial prompt.

        Args:
            prompt: The initial text prompt to start the generation.
            max_tokens: The maximum number of tokens to generate. Default is 500.
            eval_every_n: Number of tokens between yielding intermediate generated text. Default is 1.
            verbose: If True, prints the tokens per second during generation.
            **sampling_kwargs: Additional keyword arguments to control the sampling strategy.
                - temperature: Sampling temperature to control the randomness of predictions. Default is 1.0.
                - top_p: Nucleus sampling probability. If specified, the sampling will be restricted to the top p cumulative probability mass.
                - min_p: Minimum probability sampling. If specified, tokens with a probability lower than this threshold will be discarded.
                - min_tokens_to_keep: The minimum number of tokens to keep during `min_p` sampling. Only valid when `min_p` is specified.

        Yields:
            str: Generated text at intervals specified by eval_every_n.
        """
        prompt_ids = mx.array(self.tokenizer.tokenize([prompt])[0])
        tokens = []
        for n_tokens, token in enumerate(
            self.generate_tokens(prompt_ids, max_tokens, verbose, **sampling_kwargs)
        ):
            tokens.append(token)
            if (n_tokens + 1) % eval_every_n == 0:
                yield "".join(self.tokenizer.detokenize(tokens))
                tokens = []
        if tokens:
            yield "".join(self.tokenizer.detokenize(tokens))

    def generate_tokens(
        self,
        prompt_ids,
        max_tokens=500,
        verbose=False,
        **sampling_kwargs,
    ):
        """Generates tokens given the tokenized prompt.

        Args:
            prompt_ids: Array of tokenized prompt IDs.
            max_tokens: The maximum number of tokens to generate. Default is 1000.
            verbose: If True, prints the tokens per second during generation.
            **sampling_kwargs: Additional keyword arguments to control the sampling strategy.
                - temperature: Sampling temperature to control the randomness of predictions. Default is 1.0.
                - top_p: Nucleus sampling probability. If specified, the sampling will be restricted to the top p cumulative probability mass.
                - min_p: Minimum probability sampling. If specified, tokens with a probability lower than this threshold will be discarded.
                - min_tokens_to_keep: The minimum number of tokens to keep during `min_p` sampling. Only valid when `min_p` is specified.

        Yields:
            int: Generated token IDs.
        """
        if verbose:
            start_time = time.time()

        logits, state = self.prefill(prompt_ids.reshape(1, -1))
        y = sample(logits, **sampling_kwargs)

        mx.async_eval(y)

        if verbose:
            mx.eval(y)
            elapsed_time_prefill = time.time() - start_time
            nr_tokens_prompt = len(prompt_ids)

        def _step(y):
            nonlocal state
            logits, state = self.step(y, state)
            return sample(logits, **sampling_kwargs)

        next_y = _step(y)
        mx.async_eval(next_y)

        yield y.item()

        if verbose:
            start_time = time.time()

        y = next_y

        for n_tokens in range(max_tokens - 1):
            y = next_y
            next_y = _step(y)
            mx.async_eval(next_y)
            if y.item() == self.eos:
                break
            yield y.item()

        if verbose:
            elapsed_time = time.time() - start_time
            print("\n" + "-" * 50)
            print(
                f"Prompt: {nr_tokens_prompt} tokens, {nr_tokens_prompt / elapsed_time_prefill:.2f} tokens-per-sec"
            )
            print(f"Generation: {n_tokens} tokens, {n_tokens / elapsed_time:.2f} tokens-per-sec")

            peak_mem = mx.metal.get_peak_memory() / 2**30
            print(f"Peak memory: {peak_mem:.3f} GB")

    def prefill(self, x):
        """Prefills the model with given prompt."""
        x = self.embedding.encode(x)
        x, state = self.model(x)
        x_last = x[:, -1, :]
        x_last = self.head(x_last)
        return x_last, state

    def step(self, x, state=None):
        """Autoregressive step for generating model response."""
        x = self.embedding.encode(x)
        x, state = self.model.step(x, state=state)
        x = self.head(x)
        return x, state


def sample(logits, **sampling_kwargs):
    """Sample from the logits using the specified sampling strategy."""
    if "min_p" in sampling_kwargs:
        allowed_keys = {"min_p", "min_tokens_to_keep", "temperature"}
        assert set(sampling_kwargs.keys()).issubset(
            allowed_keys
        ), f"keys {sampling_kwargs.keys()} not in {allowed_keys}"
        return min_p_sampling(logits, **sampling_kwargs)
    elif "top_p" in sampling_kwargs:
        allowed_keys = {"top_p", "temperature"}
        assert set(sampling_kwargs.keys()).issubset(
            allowed_keys
        ), f"keys {sampling_kwargs.keys()} not in {allowed_keys}"
        return top_p_sampling(logits, **sampling_kwargs)
    else:
        allowed_keys = {"temperature"}
        assert set(sampling_kwargs.keys()).issubset(
            allowed_keys
        ), f"keys {sampling_kwargs.keys()} not in {allowed_keys}"
        return categorical_sampling(logits, **sampling_kwargs)
