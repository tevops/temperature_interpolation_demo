# OBTAINING THE DATA
import shutil
import pandas as pd

from git import Repo
from pathlib import Path
from typing import List

from temperature_calculator.utils import logger
from temperature_calculator.utils.constants import (
    room_temp_repo_path,
    et_temp_repo_path,

    ROOM_TEMP_REPO_URL,
    ET_TEMP_REPO_URL,

    room_temp_csv_path,
    et_temp_csv_path,

    TIME, TEMPERATURE, rootdir,
)


def download_room_temp_dataset() -> Path:
    shutil.rmtree(
        room_temp_repo_path.as_posix(),
        ignore_errors=True,
    )
    Repo.clone_from(
        ROOM_TEMP_REPO_URL,
        room_temp_repo_path.as_posix()
    )

    assert room_temp_csv_path.exists()
    return room_temp_csv_path


def download_et_temp_dataset() -> Path:
    shutil.rmtree(
        et_temp_repo_path.as_posix(),
        ignore_errors=True,
    )
    Repo.clone_from(
        ET_TEMP_REPO_URL,
        et_temp_repo_path.as_posix()
    )
    assert et_temp_csv_path.exists()
    return et_temp_csv_path


def load_room_temperature_data(
        columns: List = [TIME, TEMPERATURE]
) -> pd.DataFrame:
    logger.info(f"LOADING DATA FROM  {room_temp_csv_path.relative_to(rootdir)}")
    df = pd.read_csv(room_temp_csv_path)
    return df.query("room=='Room 1'")[columns]


def load_et_temperature_data():
    logger.info(f"LOADING DATA FROM  {et_temp_csv_path.relative_to(rootdir)}")
    df = pd.read_csv(et_temp_csv_path)
    return df.rename(
        columns={
            "date": TIME,
            "OT": TEMPERATURE,
        }
    ).drop(
        columns=[
            'HUFL',
            'HULL',
            'MUFL',
            'MULL',
            'LUFL',
            'LULL',
        ]
    )