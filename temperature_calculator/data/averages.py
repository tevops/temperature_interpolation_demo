import pandas as pd
import functools

from typing import Dict, List

from temperature_calculator.data.preprocessing import (
    generate_non_equidistant_sample,
    get_random_timestamps,
    get_timespan,
)
from temperature_calculator.utils import logger
from temperature_calculator\
    .utils\
    .constants import (
        TEMPERATURE,
        REAL_MEAN,
        RESULT,
        TIME,
    )


def add_results(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        kwargs[RESULT][func.__name__] = func(*args, **kwargs)
        return kwargs[RESULT]

    return wrap


class TemperatureCalculator:
    mdiff = "mdiff"
    wtemp = "wtemp"

    @classmethod
    @add_results
    def simple_average(
            cls,
            df: pd.DataFrame,
            temperature_column: str,
            results: Dict,
    ) -> Dict:
        assert results
        return df[temperature_column].mean()  # SIMPLE AVERAGE

    @classmethod
    @add_results
    def weighted_average(
            cls,
            df: pd.DataFrame,
            time_column: str,
            temperature_column: str,
            results: Dict
    ):
        assert results
        df[cls.mdiff] = abs(df[time_column] - df[time_column].shift(-1))
        df[cls.wtemp] = df[temperature_column] * df[cls.mdiff]
        return df[cls.wtemp].sum() / df[cls.mdiff].sum()  # WEIGHTED AVERAGE

    @classmethod
    def interpolated_average(
            cls,
            df: pd.DataFrame,
            time_column: str,
            temperature_column: str,
            results: Dict
    ):
        ...


def calculate_mean_temperatures(
        frac: float,
        df: pd.DataFrame,

        time_column: str,
        temperature_column: str,
        numeric_time_column: str,

        methods: List[str],
        verbose: bool = False,
):
    sample_dataframe = generate_non_equidistant_sample(
        time_column=numeric_time_column,
        verbose=verbose,
        frac=frac,
        df=df,

    )

    mint, maxt = get_random_timestamps(
        df=sample_dataframe,
        time_column=time_column,
        numeric_time_column=numeric_time_column,
    )

    timespan_df = get_timespan(
        mint=mint,
        maxt=maxt,
        df=sample_dataframe,
        time_column=time_column,
    )

    rmean = (
        df[
            (df[TIME] > mint) & (df[TIME] < maxt)
            ][TEMPERATURE]
        .mean()
    )  # REAL MEAN

    result = {REAL_MEAN: rmean}

    if "sa" in methods:
        TemperatureCalculator.simple_average(
            df=timespan_df,
            results=result,
            temperature_column=temperature_column,
        )  # SIMPLE MEAN

    if "wa" in methods:
        TemperatureCalculator.weighted_average(
            df=timespan_df,
            results=result,
            temperature_column=temperature_column,

            time_column=numeric_time_column,
        )  # WEIGHTED MEAN

    if "ia" in methods:
        TemperatureCalculator.interpolated_average(
            df=timespan_df,
            results=result,
            temperature_column=temperature_column,

            time_column=numeric_time_column,
        )  # INTERPOLATED MEAN

    if verbose:
        print(
            f"AVERAGE TEMPERATURE BETWEEN {mint} AND {maxt}",
            '\n\t - SIMPLE\t', result.get(TemperatureCalculator.simple_average.__name__),
            '\n\t - WEIGHTED\t', result.get(TemperatureCalculator.weighted_average.__name__),
            '\n\t - REAL VALUE\t', result.get(REAL_MEAN),
        )

    return result
