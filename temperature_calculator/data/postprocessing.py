import pandas as pd

from matplotlib import pyplot as plt

from temperature_calculator\
    .data\
    .averages import TemperatureCalculator as TC
from temperature_calculator.utils.constants import (
    DIFF_REAL_SIMPLE, 
    DIFF_REAL_WEIGHTED, 
    REAL_MEAN,
)


def calculate_results(results):
    rdf = pd.DataFrame(results)
    rdf[DIFF_REAL_SIMPLE] = abs(rdf[TC.simple_average.__name__] - rdf[REAL_MEAN])
    rdf[DIFF_REAL_WEIGHTED] = abs(rdf[TC.weighted_average.__name__] - rdf[REAL_MEAN])
    return rdf


def summarize_results(results):
    rdf = calculate_results(results)
    print("# DIFFERENCE BETWEEN SIMPLE MEAN AND REAL MEAN\n")
    print(rdf[DIFF_REAL_SIMPLE].describe())
    print("- " * 30)
    print("# DIFFERENCE BETWEEN WEIGHTED MEAN AND REAL MEAN\n")
    print(rdf[DIFF_REAL_WEIGHTED].describe())
    rdf[
        [
            TC.simple_average.__name__,
            TC.weighted_average.__name__,
            REAL_MEAN
        ]
    ].plot.kde(alpha=0.3)
    plt.show()
