from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_data(data_folder: str):
    paths = list(Path(data_folder).glob('*.txt'))
    dfs = []

    for path in tqdm(paths, desc='Loading data'):
        df = pd.read_csv(path, sep='\t', na_values=['None', '.'])

        dfs.append(df)

    return dfs
