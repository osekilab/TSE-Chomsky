#!/bin/bash
# generate data
python scripts/generate_data.py an_bn \
    --num_of_terms $num_term \
    --min_iter 2 \
    --max_iter 15 \
    --max_iter_test 50 \
    --erase_duplicated_data true \
    --dist "uniform"

# train/predict
poetry run python scripts/train_lm.py \
    --config configs/gpt2.json \
    --task "AnBn" \
    --batch_size 512 \
    --eval_batch_size 512 \
    --num_epochs 15 \
    --num_terms $term \
    --num_of_nonterms 3 \
    --min_iter 2 \
    --max_iter 15 \
    --max_iter_test 50 \
    --dist uniform \
    --num_workers  1 \
    --exp_name test \
    --learning_rate $learning_rate \
    --erase_duplicated_data \
    --seed 0 \
    --accelerator "gpu" \
    --devices 1 \
    --val_check_interval 0.5
