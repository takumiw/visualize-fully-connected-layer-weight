from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def get_callback(
    path_chpt: str,
    patience: int = 30,
) -> List[Any]:
    """callback settinngs
    Args:
        path_chpt (str): path to save checkpoint
    Returns:
        callbacks (List[Any]): List of Callback
    """
    callbacks = []
    callbacks.append(
        EarlyStopping(
            monitor="val_loss", min_delta=0, patience=patience, verbose=1, mode="min"
        )
    )
    callbacks.append(ModelCheckpoint(filepath=path_chpt, save_best_only=True))
    return callbacks
