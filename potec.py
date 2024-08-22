from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class Potec(Dataset):

    def __init__(self, potec_repo_root: str):
        self.repo_root = Path(potec_repo_root)

        self.scanpath_folder = self.repo_root / 'eyetracking_data/scanpaths/'
        self.merged_scanpaths_folder = self.repo_root / 'eyetracking_data/scanpaths_merged/'
        self.reading_measures_folder = self.repo_root / 'eyetracking_data/reading_measures/'
        self.merged_reading_measures_folder = self.repo_root / 'eyetracking_data/reading_measures_merged/'

    def load_potec_merged_scanpaths(self, label_name: str) -> (list, pd.DataFrame):
        # sort to make sure we have the same order
        paths = sorted(list(self.merged_scanpaths_folder.glob('*.tsv')))
        dfs = []

        reader_ids = []
        text_ids = []
        labels = []

        for path in tqdm(paths, desc='Loading data'):
            df = pd.read_csv(path, sep='\t', na_values=['None', '.'])

            reader_ids.append(df['reader_id'].iloc[0])
            text_ids.append(df['text_id'].iloc[0])

            if label_name == 'expert_cls_label':
                label = df['expert_reading_label_numeric'].iloc[0]

            elif label_name == 'expert_status':
                label = df['level_of_studies_numeric'].iloc[0]

            labels.append(label)

            dfs.append(df)

        # save the sample mapped to unique data identifiers to make sure it is reproducible
        sample_mapping = pd.DataFrame(
            {'sample_id': range(len(reader_ids)), 'reader_id': reader_ids, 'text_id': text_ids, 'label': labels})

        return dfs, np.array(labels), sample_mapping
