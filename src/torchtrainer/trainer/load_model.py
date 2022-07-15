from os import PathLike

import dill
import torch

__all__ = ["load_best_model"]


def load_best_model(f: PathLike, map_location: str = "cpu") -> torch.nn.Module:
    """
    Load the best model from a file.
    """
    return torch.load(f, map_location=map_location, pickle_module=dill)
