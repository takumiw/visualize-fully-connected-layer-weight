import numpy as np
from tensorflow.keras.models import Model

def get_features(model: Model, X: np.ndarray) -> np.ndarray:
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model.predict(X, verbose=0)
    return features