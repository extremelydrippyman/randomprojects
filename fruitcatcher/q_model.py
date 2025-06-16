import tensorflow as tf
from tensorflow.python.util import lazy_loader

keras = lazy_loader.LazyLoader("keras", globals(), "keras")

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(24, input_shape=(3,), activation = 'relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(3, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse'
    )

    return model