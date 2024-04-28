from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)

from temperature_calculator.utils import logger


def create_model(input_shape: int):
    logger.info(f"CREATING AN LSTM-AUTOENCODER WITH INPUT LEN {input_shape}")
    model = Sequential()

    model.add(
        LSTM(
            100,
            activation='relu',
            input_shape=(input_shape, 1)
        )
    )

    model.add(RepeatVector(input_shape))
    model.add(
        LSTM(
            100,
            activation='relu',
            return_sequences=True,
        )
    )

    model.add(TimeDistributed(Dense(1)))
    model.compile(
        optimizer='Adam',
        loss='mse',
    )
    logger.info(f"\tRETURNING THE AUTOENCODER")

    return model

