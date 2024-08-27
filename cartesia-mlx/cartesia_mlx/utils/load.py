import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from cartesia_mlx.utils.configure import instantiate


class ModelNotFoundError(Exception):
    """Exception raised when the model is not found."""

    def __init__(self, message):
        """
        Args:
            message: The error message.
        """
        self.message = message
        super().__init__(self.message)


def get_model_path(
    path_or_hf_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
) -> pathlib.Path:
    """Get the local path of the model."""
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=["*.safetensors"],
            )
        )
        model_path = next(model_path.glob("*.safetensors"), None)
    if model_path is None:
        raise ValueError(f"No model path found in '{path_or_hf_repo}'.")
    return model_path


def instantiate_from_file(file_path) -> nn.Module:
    """Instantiates a model from the safetensors checkpoint file.

    Args:
        file_path: The path to the safetensors checkpoint file.

    Returns:
        The mlx module.
    """
    state_dict, cfg = mx.load(str(file_path), return_metadata=True)
    cfg = [(k, json.loads(v)) for k, v in cfg.items()]
    cfg = tree_unflatten(cfg)
    model = instantiate(cfg)
    state_dict_nested = tree_unflatten(list(state_dict.items()))
    model.update(state_dict_nested)
    mx.eval(model.parameters())
    model.eval()
    return model


def from_pretrained(
    path_or_hf_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
):
    """Load a model from a local path or huggingface repository.

    Args:
        path_or_hf_repo: The path to the model or the huggingface repository.
        revision: The revision of the model.
    """
    model_path = get_model_path(path_or_hf_repo, revision)
    if not model_path:
        logging.error(f"No safetensors file found in {model_path}")
        raise FileNotFoundError(f"No safetensors file found in {model_path}")
    model = instantiate_from_file(model_path)
    return model
