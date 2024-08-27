import copy
import re


def filter_startswith(dictionary: dict, prefix: str, invert: bool = False):
    """Returns the subtree of a flattened dictionary that starts with a given prefix."""
    filtered_dict = copy.deepcopy(dictionary)
    pattern = re.compile(re.escape(prefix) + ".*")

    def matches_pattern(key):
        if invert:
            return not pattern.match(key)
        else:
            return pattern.match(key)

    filtered_dict = {key: value for key, value in filtered_dict.items() if matches_pattern(key)}
    return filtered_dict


def filter_startswith_strip(dictionary: dict, prefix: str):
    """Returns the subtree of a flattened dictionary that starts with a given prefix and strips the prefix from the keys."""
    prefix_len = len(prefix)
    filtered_dict = copy.deepcopy(dictionary)
    filtered_dict = filter_startswith(filtered_dict, prefix, invert=False)
    filtered_dict = {k[prefix_len:]: v for k, v in filtered_dict.items()}
    return filtered_dict
