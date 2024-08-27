"""
This is a wrapper for the lm-evaluation-harness that enables evaluating the Rene model.
Other standard models can still be evaluated with this script.
The command-line interface is the same as that of the standard lm-evaluation-harness.
To evaluate a Rene-class model, pass `rene_ssm` to the `--model` argument.
"""

from typing import Optional, Union

import lm_eval.models.utils
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("rene_ssm")
class ReneLMWrapper(HFLM):
    """Wrapper for Rene model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            # rene currently only supports causal models
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "allenai/OLMo-1B-hf"),
            max_length=kwargs.pop("max_length", 4096),
            **kwargs,
        )

    def _get_config(self, pretrained: str, **kwargs) -> None:
        from cartesia_pytorch.configuration_rene import ReneConfig

        self._config = ReneConfig.from_pretrained(pretrained)

    def _create_model(
        self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs
    ) -> None:
        from cartesia_pytorch.rene import ReneLMHeadModel

        self._model = ReneLMHeadModel.from_pretrained(
            pretrained,
            device=self._device,
            dtype=torch.float16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # The custom GenerationMixin imported from mamba_ssm currently does not support
        # passing stopping criteria.
        # For the time being, we simply generate to max length, then truncate (equivalent result).
        # This should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, 1, context.shape[0])

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            # stopping_criteria=stopping_criteria,
            # pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=True,
            **generation_kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
