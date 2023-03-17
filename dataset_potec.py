from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = 'C:\Users\debor\data\PotsdamTextbookCorpus\OSF\eyetracking_data\merged'


# Define a custom dataset class
class PoTeC(Dataset):
    def __init__(self, path):
        self.data_folder_path = path

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def read_data_files(self):
        paths_merged_files = Path(self.data_folder_path).glob('*.txt')

        first_path_fix = next(paths_merged_files)
        data_tsv_fix = pd.read_csv(first_path_fix, sep='\t', na_values=['None'])





