from pathlib import Path

import pandas as pd

from extract_features_rf import get_combined_features


DATA_FOLDER_PATH = '/Users/debor/repos/PoTeC-data/eyetracking_data/scanpaths_reader_rm_wf'

paths = list(Path(DATA_FOLDER_PATH).glob('*.txt'))[:3]

dfs = []

for path in paths:
    df = pd.read_csv(path, sep='\t', na_values=['.'])

    dfs.append(df)

data_arr, feature_names = get_combined_features(dfs)

