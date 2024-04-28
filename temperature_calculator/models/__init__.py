from pathlib import Path
from typing import Dict

from keras.models import model_from_json

from temperature_calculator.utils import logger
from temperature_calculator.utils.constants import (
    MODEL_PATH,
    rootdir,
)


def save_model(model, path_model: Path, path_weights: Path):
    logger.info(f"SAVING MODEL IN {path_model.relative_to(rootdir)}")
    path_model.parent.mkdir(
        exist_ok=True,
        parents=True,
    )

    with open(path_model, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(path_weights.as_posix())


def load_model(path_model: Path, path_weights: Path):
    logger.info(f"LOADING MODEL FROM {path_model.relative_to(rootdir)}")
    with open(path_model, 'r') as json_model_file:
        loaded_model_json = json_model_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights.as_posix())
    loaded_model.compile(
        loss='mse',
        optimizer='Adam',
    )
    return loaded_model


def get_model_path(name: str) -> Dict:
    return {
        "path_model": (
            MODEL_PATH
            .joinpath(name)
            .with_suffix(".json")
        ),
        "path_weights": (
            MODEL_PATH
            .joinpath(name)
            .with_suffix(".h5")
        )

    }
