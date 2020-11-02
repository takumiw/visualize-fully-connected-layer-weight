from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def get_features(model: Model, X: np.ndarray) -> np.ndarray:
    """Calculate features of VGG8"""
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model.predict(X, verbose=0)
    return features


def get_features_arcface(model: Model, X: np.ndarray) -> np.ndarray:
    """Calculate features of ArcFace"""
    model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    features = model.predict(X, verbose=0)
    return features


def plot_model(model: Model, path: str = "model.png"):
    return tf.keras.utils.plot_model(model, to_file=path, show_shapes=True)


def predict_arcface(model: Model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    W = model.get_layer("arc_face").W
    arcface_model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    arcface_features = arcface_model.predict(X, verbose=0)
    arcface_features /= np.linalg.norm(arcface_features, axis=1, keepdims=True)
    pred = K.eval(arcface_features @ W)
    pred_prob = K.softmax(pred)
    return pred_prob, tf.math.argmax(pred_prob, axis=1)
