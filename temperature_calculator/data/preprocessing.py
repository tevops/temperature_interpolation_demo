from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from temperature_calculator.utils import logger
from temperature_calculator.utils.constants import (
    SEQUENCE_ID,
    TEMPERATURE,
    NEXT_DIFF,
    SEQUENCE,
    LENSEQ,
    SPLSEQ,
    HOURS, MINUTES,
)


def convert_timestamp(
        df: pd.DataFrame,
        time_column_source,
        time_column_target,
        verbose: bool,
):
    if verbose:
        logger.info("CONVERTING TIMESTAMP TO NUMERIC")
    converted = (
        df
        [time_column_source]
        .apply(
            lambda t: int(
                datetime
                .strptime(t, '%Y-%m-%d %H:%M:%S')
                .strftime('%s')
            )
        )
    )
    if time_column_target == MINUTES:
        df[time_column_target] = converted / 60
    elif time_column_target == HOURS:
        df[time_column_target] = converted / 3600
    return df


def generate_non_equidistant_sample(
        df: pd.DataFrame,
        verbose: bool,
        time_column: str,
        frac: float = 0.04,
):
    sdf = df.sample(frac=frac).sort_values(time_column)

    if verbose:
        logger.info("GENERATING NON-EQUIDISTANT SAMPLE")
        dist = "|\tGAP\t|\tCOUNT\n" + "-"*30+"\n"+"\n".join(
            [
                f"|\t{k}\t|\t{v}" for k, v in dict(
                    sdf[time_column]
                    .diff()
                    .value_counts()
                ).items()
            ]
        )

        logger.info("DISTRIBUTION OF TIME GAPS\n" + dist)

    return sdf


def get_random_timestamps(df, time_column: str, numeric_time_column: str, verbose: bool = False):
    _tt = (
        df
        .sample(2)
        .sort_values(numeric_time_column)
    )
    mint, maxt = _tt[time_column].values[0], _tt[time_column].values[1]
    if verbose:
        logger.info(f"RANDOM TIMESPAN START: {mint} END: {maxt}")
    return mint, maxt


def get_timespan(
        df: pd.DataFrame,
        time_column: str,
        mint: str,
        maxt: str,
) -> pd.DataFrame:
    return df[
        (
                df[time_column] > mint
        ) & (
                df[time_column] < maxt
        )
        ].copy(deep=True)


def extract_continuous_sequences(df: pd.DataFrame, time_column: str, verbose: bool) -> pd.DataFrame:
    if verbose:
        logger.info("EXTRACTING CONTINUOUS SEQUENCES FROM NON-EQUIDISTANT DATASET")
    df[NEXT_DIFF] = abs(df[time_column].diff(-1)) - 1
    df[SEQUENCE] = (df[NEXT_DIFF].shift(-1) == 0) | (df[NEXT_DIFF] == 0)

    sequence_df = df[df[SEQUENCE] == True].copy(deep=True)
    sequence_df[SEQUENCE_ID] = sequence_df[NEXT_DIFF].cumsum()
    return sequence_df


def create_trainable_data(
        sequences: pd.DataFrame,
        verbose: bool,
        min_lenght: int = 2,
        time_column: str = HOURS,
) -> pd.DataFrame:

    csequences = extract_continuous_sequences(
        time_column=time_column,
        verbose=verbose,
        df=sequences,
    )
    summary = (
        csequences
        .groupby(SEQUENCE_ID)[TEMPERATURE]
        .apply(len)
        .reset_index()
        .rename(columns={TEMPERATURE: LENSEQ})
    )

    df = (
        csequences[
            csequences.SEQUENCE_ID.isin(
                summary[summary[LENSEQ] > min_lenght]
                [SEQUENCE_ID]
                .values
            )
        ].groupby(SEQUENCE_ID)
        [TEMPERATURE]
        .apply(list)
    ).reset_index()
    return df.merge(summary, on=SEQUENCE_ID)


def flatten_list(biglist: List):
    return [item for sublist in biglist for item in sublist]


def split(sequence_: Union[List, np.array], sublen: int) -> np.array:
    return [
        np.array(
            [
                sequence_[ix:ix + sublen]
            ]
        ).reshape((sublen, 1)) for ix in range(
            0,
            (len(sequence_) - sublen + 1),
            1
        )
    ]


def pad(sequence_: Union[List, np.array], sublen: int) -> np.array:
    return [np.concatenate(
        [
            -10*np.ones((sublen - len(sequence_))),
            sequence_
        ]
    ).reshape((sublen, 1))]


def mask(sequence_: np.array, frac: float, len_: int) -> np.array:
    return np.where(
        np.random.choice(
            [True, False],
            size=(len_, 1),
            p=[frac, 1-frac],  # better probabilities can be thought of
        ),
        sequence_,
        -10,
    )


def split_row(row, len_: int = 9):
    """ Carries out padding/splitting. """
    if row[LENSEQ] > len_:
        return split(
            sequence_=row[TEMPERATURE],
            sublen=len_,
        )
    return pad(
        sequence_=row[TEMPERATURE],
        sublen=len_,
    )


def extract_sequences(df: pd.DataFrame, len_: int):
    logger.info("APPLYING INPUT STANDARDIZATION ")
    df[SPLSEQ] = df.apply(
        lambda row: split_row(row, len_=len_),
        axis=1
    )
    return df
