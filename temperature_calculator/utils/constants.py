from pathlib import Path

# # DATA-SOURCE RELATED CONSTANTS
rootdir = Path(__file__).parents[2]
dataset = rootdir.joinpath('datasets')
room_temp_repo_path = dataset.joinpath("room_temperatures")
room_temp_csv_path = (
    room_temp_repo_path
    .joinpath("temperatures")
    .with_suffix(".csv")
)

ROOM_TEMP_REPO_URL = "https://github.com/johnmyleswhite/room_temperatures"

et_temp_repo_path = dataset.joinpath("ETDataset")
et_temp_csv_path = (
    et_temp_repo_path
    .joinpath("ETT-small")
    .joinpath("ETTh1")
    .with_suffix(".csv")
)

ET_TEMP_REPO_URL = "https://github.com/zhouhaoyi/ETDataset"

# # DATA-PROCESSING RELATED CONSTANTS
TIME = "time"
HOURS = "hours"
MINUTES = 'minutes'
TEMPERATURE = "temperature"
PREDICTED = TEMPERATURE + "_predicted"
INTERPOLATED = TEMPERATURE + "_interpolated"

REAL_MEAN = "rmean"
DIFF_REAL_SIMPLE = "diff_rmean_smean"
DIFF_REAL_WEIGHTED = "diff_rmean_wmean"
DIFF_REAL_INTERPOL = "diff_rmean_imean"

# # MODELLING RELATED CONSTANTS
NEXT_DIFF = "NEXT_DIFF"

SPLSEQ = "split_sequences"
LENSEQ = "len_sequence"
SEQUENCE = "SEQUENCE"
SEQUENCE_ID = "SEQUENCE_ID"

RESULT = "results"
MODEL_PATH = rootdir.joinpath("models")


