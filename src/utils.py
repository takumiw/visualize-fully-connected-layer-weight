import matplotlib.pyplot as plt


def plot_learning_history(fit, metric: str = "accuracy", path: str = "history.png"):
    """Plot learning curve
    Args:
        fit: History object
        path (str, default="history.png")
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

    return fig

    fig.savefig(path)
    plt.close()
