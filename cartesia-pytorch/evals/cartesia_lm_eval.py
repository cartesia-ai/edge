"""
This is a wrapper for the lm-evaluation-harness that enables evaluating Llamba and Rene models.
Other standard models can still be evaluated with this script.
The command-line interface is the same as that of the standard lm-evaluation-harness.
To evaluate a Rene-class model, pass `rene_ssm` to the `--model` argument.
"""

from lm_eval.__main__ import cli_evaluate

from ..cartesia_pytorch.Llamba.llamba_eval_wrapper import LlambaLMWrapper  # noqa: F401
from ..cartesia_pytorch.Rene.rene_eval_wrapper import ReneLMWrapper  # noqa: F401

if __name__ == "__main__":
    cli_evaluate()
