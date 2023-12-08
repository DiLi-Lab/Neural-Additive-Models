
python3 nam_train.py \
    --dataset_folder "/Users/debor/repos/PoTeC-data" \
    --dataset_name "PoTeC" \
    --training_epochs 30 \
    --batch_size 32 \
    --logdir "logs/nam_cv" \
    --group_by "reader_id" \
    --all_folds true

