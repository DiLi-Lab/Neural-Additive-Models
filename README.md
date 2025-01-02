# PoTeC expert level prediction
## Get data
```bash
git clone git@github.com:dili-lab/PoTeC PoTeC-data
cd PoTeC-data
```

```bash
python download_data_files.py
```

took ~7 minutes

## Training
```bash
python nam_train.py --hp_tuning --dataset_folder PoTeC-data --data_split 1
```

## Evaluation

## Results
