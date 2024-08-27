import collections
import copy
import importlib
from typing import Any, Dict, Optional

from mlx.utils import tree_flatten, tree_unflatten

from cartesia_mlx.utils.filter import filter_startswith_strip


def set_cfg(obj, cfg, parent=None, instantiate_children=False):
    """Sets the configuration and optionally instantiates child objects.

    Args:
        cfg: Configuration dictionary to update the base configuration with.
        parent: Parent object to inherit configuration values from (optional).
        instantiate_children: Whether to instantiate child objects specified in the configuration (default is False).

    Returns:
        dict: The updated configuration dictionary.
    """
    cfg = _resolve_inheritance(copy.deepcopy(cfg))
    this_cfg = _resolve_inheritance(copy.deepcopy(obj.base_cfg))
    this_cfg = _recursive_update(this_cfg, cfg)
    this_cfg = _parent_update(this_cfg, parent)
    obj.cfg = this_cfg
    for k, v in obj.cfg.items():
        if isinstance(v, dict) and instantiate_children is True and "_class_" in v:
            setattr(obj, k, instantiate(v, parent=obj))
        else:
            setattr(obj, k, v)
    return cfg


class Inherit:
    """Inherit class for inheriting values from parent config."""

    def __init__(self, default: Any = None, from_key: Optional[str] = None):
        self.default = default
        self.from_key = from_key

    def __repr__(self):
        return f"Inherit(default={self.default}, from_key={self.from_key})"


def sub_cfg(base_cfg: Dict[str, Any], **kwargs):
    """Overwrites the base_cfg with the kwargs provided."""
    cfg = copy.deepcopy(base_cfg)
    cfg.update(kwargs)
    return cfg


def class_from_path(path: str, package_name="cartesia_mlx"):
    """Retrives class from path to the class."""
    path = package_name + "." + path
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def instantiate(cfg: Dict[str, Any], parent: Optional[Any] = None):
    """Instantiates a class from the config."""
    this_cfg = copy.deepcopy(cfg)
    assert "_class_" in this_cfg, f"Class not found in config: {this_cfg}"
    class_path = this_cfg.pop("_class_")
    cls = class_from_path(class_path)
    instance = cls(cfg=this_cfg, parent=parent)
    return instance


def _resolve_inheritance(cfg):
    """Removes all Inheritance objects from the configuration and by replacing them with inherited values."""

    def resolve_value(flat_cfg, key):
        value = flat_cfg[key]
        if isinstance(value, Inherit):
            # Replace the key with the key to inherit from
            inherit_key = key.split(".")[-1] if value.from_key is None else value.from_key

            full_inherit_key = key.split(".")

            # No parent key to inherit from -> return default value
            if len(full_inherit_key) < 2:
                return value.default

            # If the second last key is an integer, remove it (this skips the lists in inheritance)
            if full_inherit_key and _can_convert_to_int(full_inherit_key[-2]):
                full_inherit_key.pop(-2)

            # Build the full key to inherit from
            full_inherit_key = full_inherit_key[:-2]

            # Build final key to inherit from
            full_inherit_key.append(inherit_key)
            full_inherit_key = ".".join(full_inherit_key)

            # If the key to inherit from exists, return its value
            if full_inherit_key in flat_cfg:
                return flat_cfg[full_inherit_key]

            # Else, check if inheriting a whole subtree is possible
            elif sub_tree := filter_startswith_strip(flat_cfg, full_inherit_key + "."):
                return sub_tree

            # Otherwise, return the default value
            return value.default
        return value

    # Flatten the configuration dictionary
    flat_cfg = dict(tree_flatten(copy.deepcopy(cfg)))

    # Sort the keys by the number of nodes (levels) in the key
    sorted_keys = sorted(flat_cfg.keys(), key=lambda k: len(k.split(".")))

    # Resolve inheritance in the sorted order
    for key in sorted_keys:
        flat_cfg[key] = resolve_value(flat_cfg, key)

    # Unflatten the dictionary back to the original nested structure

    resolved_cfg = tree_unflatten(list(flat_cfg.items()))
    return resolved_cfg


def _recursive_update(d: Dict[str, Any], u: Dict[str, Any]):
    if u is None:
        return d
    for k, v in u.items():
        t = d.get(k, {})
        if isinstance(v, collections.abc.Mapping) and t is not None:
            d[k] = _recursive_update(t, v)
        else:
            d[k] = v
    return d


def _parent_update(this_cfg: Dict[str, Any], parent: Optional[Any]):
    for k, v in this_cfg.items():
        if isinstance(v, Inherit):
            kp = v.from_key if v.from_key is not None else k
            if parent is not None and hasattr(parent, kp):
                this_cfg[k] = copy.deepcopy(getattr(parent, kp))
            elif parent is None and v.default is not None:
                this_cfg[k] = copy.deepcopy(v.default)
            elif parent is not None and not hasattr(parent, kp):
                this_cfg[k] = copy.deepcopy(v.default)
            else:
                raise ValueError(
                    f"Inhertitance error: Parent attribute {k} not found and no default provided."
                )
    return this_cfg


def _can_convert_to_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False
