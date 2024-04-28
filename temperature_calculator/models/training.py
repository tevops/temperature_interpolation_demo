import numpy as np
import pandas as pd

from typing import (
    Tuple,
    List,
    Dict,
)

from sklearn.model_selection import train_test_split

from temperature_calculator.utils import logger
from temperature_calculator.utils.constants import (
    SPLSEQ,
)

from temperature_calculator \
    .data \
    .preprocessing import (
    mask,
    extract_sequences,
    create_trainable_data,
)

from temperature_calculator \
    .models \
    .sequential import create_model


def create_masked_dataframe(num_resample: int, data: List, frac: float, len_: int) -> Tuple:
    df_input = []
    df_output = []
    for item in data:
        for ix in range(num_resample):
            df_input.append(
                mask(
                    sequence_=item,
                    frac=frac,
                    len_=len_,

                )
            )
            df_output.append(item)
    return df_input, df_output


def check_train_data(train_data: List):
    assert set(map(type, train_data)) == {np.ndarray}
    assert set(map(len, train_data)) == {9}
    assert set(map(lambda r: r.shape, train_data)) == {(9, 1)}


def create_train_data(df: pd.DataFrame, len_: int) -> List:
    logger.info("CREATING TRAIN DATA")
    sequence_df = extract_sequences(df=df, len_=len_)
    train_data = []
    for ix, row in sequence_df.iterrows():
        train_data.extend(row[SPLSEQ])
    check_train_data(train_data)
    return train_data


def process_data_for_training(df: pd.DataFrame, verbose: bool, len_: int, frac: float) -> Dict:
    train_df = create_trainable_data(sequences=df, verbose=verbose)
    x_train, x_test = train_test_split(
        create_train_data(
            df=train_df,
            len_=len_,
        )
    )

    (
        x_train_masked_input,
        x_train_non_masked_output,
    ) = create_masked_dataframe(data=x_train, num_resample=5, frac=frac, len_=len_)
    (
        x_val_masked_input,
        x_val_non_masked_output,
    ) = create_masked_dataframe(data=x_test, num_resample=5, frac=frac, len_=len_)
    return dict(
        map(
            lambda kv: (kv[0], np.array(kv[1])),
            {
                "x_train_input": x_train_masked_input,
                "x_train_output": x_train_non_masked_output,
                "x_val_input": x_val_masked_input,
                "x_val_output": x_val_non_masked_output,

            }.items()
        )
    )


def train_model(
        input_shape: int,
        x_train_input: np.array,
        x_train_output: np.array,
        x_val_input: np.array,
        x_val_output: np.array,
        batch_size: int = 3,
        num_epochs: int = 10,
):
    logger.info(
        f"PREPARING TO TRAIN A MODEL ON "
        f"DATA WITH SHAPE {x_train_input.shape}"
    )

    model = create_model(input_shape=input_shape)
    logger.info("TRAINING STARTED")
    model.fit(
        x_train_input,
        x_train_output,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(
            np.array(x_val_input),
            np.array(x_val_output),
        )
    )
    logger.info("TRAINING FINISHED")
    return model
