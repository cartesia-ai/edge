import warnings

from transformers import AutoTokenizer

from cartesia_mlx.utils.configure import set_cfg

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)


class Tokenizer:
    """Tokenizer Wrapper for Huggingface AutoTokenizer."""

    base_cfg = dict(
        _class_="utils.tokenizer.Tokenizer",
        name="allenai/OLMo-1B-hf",
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        self.name: str
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.tokenizer.decode([0])  # warm up

    def tokenize(self, x: list) -> list:
        """Tokenize list of strings."""
        tokens = self.tokenizer(x).input_ids
        return tokens

    def detokenize(self, tokens: list) -> list:
        """Detokenize list of tokens."""
        return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
