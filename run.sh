#!/bin/bash
# python run_prediction.py

python eval.py \
    res/result_dev.tsv \
    data/dev.tsv \
    --gold_col label \
    --id_col id \
    --extract_lang_from_id >> log/eval_dev.log