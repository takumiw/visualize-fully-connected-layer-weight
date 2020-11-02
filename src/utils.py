from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_learning_history(fit, metric: str = "accuracy"):
    """Plot learning curve
    Args:
        fit: History object
        path (str, default="history.png")
    Returns:
        fig: figure of learning_history
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(fit.history["loss"], label="train")
    axL.plot(fit.history["val_loss"], label="validation")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(fit.history[metric], label="train")
    axR.plot(fit.history[f"val_{metric}"], label="validation")
    axR.set_title(metric.capitalize())
    axR.set_xlabel("epoch")
    axR.set_ylabel(metric)
    axR.legend(loc="best")
    plt.close()
    return fig


def plot_confusion_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: str = "true",
    figsize: Sequence[int] = (5, 4),
) -> np.ndarray:
    """Plot confusion matrix
    Args:
        true (numpy.array): true label
        pred (numpy.array): predicted label
        labels (List[str]), default=None] list of label names
        normalize (str, default="true): whether to normalize scores, chosen from "true" or "false"
    Returns:
        fig: figure of confusion matrix
    """
    cm = confusion_matrix(true, pred, normalize=normalize)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        square=True,
        vmin=0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix")

    plt.close()
    return fig
