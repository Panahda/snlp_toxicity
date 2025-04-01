import argparse
import json
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', quoting=3)
    return df

def calculate_standard_metrics(y_true, y_pred):
    # Assumes valid inputs
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='binary',
        pos_label=1,
        zero_division=0
    )
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(
    pred_file,
    gold_file,
    id_col,
    pred_col,
    gold_col,
):
    results = {}

    # --- Load Data ---
    pred_df = load_data(pred_file)
    gold_df = load_data(gold_file)

    pred_df_select = pred_df[[id_col, pred_col]]
    gold_df_select = gold_df[[id_col, gold_col]]
    merged_df = pd.merge(gold_df_select, pred_df_select, on=id_col, how="inner")

    y_true_all = merged_df[gold_col].astype(int)
    y_pred_all = merged_df[pred_col].astype(int)

    # --- Calculate Overall Metrics ---
    overall_metrics = calculate_standard_metrics(y_true_all, y_pred_all)
    for metric_name, value in overall_metrics.items():
        results[f"multilingual_{metric_name}"] = value

    # --- Calculate Per-Language Metrics ---
    lang_col = 'lang'
    merged_df[lang_col] = merged_df[id_col].str.split('_', n=1, expand=True)[0]
    langs = ['eng', 'ger', 'fin']
    for lang in langs:
        lang_mask = merged_df[lang_col] == lang
        y_true_lang = y_true_all[lang_mask]
        y_pred_lang = y_pred_all[lang_mask]

        # Minimal check if subset is empty after masking
        if not y_true_lang.empty:
            lang_metrics = calculate_standard_metrics(y_true_lang, y_pred_lang)
            for metric_name, value in lang_metrics.items():
                results[f"{lang}_{metric_name}"] = value
                
    return results


def main():
    pred_file = 'res/result_dev.tsv'
    gold_file = 'data/dev.tsv'
    id_col = 'id'
    pred_col = 'predicted'
    gold_col = 'label'
    
    results = evaluate(
        pred_file,
        gold_file,
        id_col,
        pred_col,
        gold_col
    )

    print(json.dumps(results, sort_keys=True))

if __name__ == "__main__":
    main()
