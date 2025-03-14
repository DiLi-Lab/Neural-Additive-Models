# Using NAMs for eye-tracking data
## Get PoTeC data
```bash
git clone git@github.com:dili-lab/PoTeC PoTeC-data
cd PoTeC-data
```

```bash
python download_data_files.py
```

took ~7 minutes

## Training and evaluating the NAM model
The model can be trained using three different labels for the three different tasks which are: expert_cls_label, all_bq_correct and all_tq_correct. 
An example call for one label including hyperparemeter tuning is shown below:
```bash
python nam_train.py --hp_tuning --label "expert_cls_label" --dataset_folder PoTeC-data
```

## Training and evaluating the baselines
The configurations for each setting are stored in a separate .json file. See below for an example call.

```bash
python evaluation.py --config "evaluation_configs/config_baseline_hp_tuning_2_labels_new-reader-split_label_expert_cls.json" --hp-tuning
```

## Analyse features
In order to extract the most important features, extract_feature_contribution.py can be run. Note that the log directory to the trained model and the label need to be provided manually at the top of the file (lines 30 and 33). The Jupyter Notebook in the folder "feature_analysis" can then be run to analyse the files and create the plots.
