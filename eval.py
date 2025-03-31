import argparse
import json
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import csv # Import csv module for QUOTE_NONE

# --- Constants ---
DEFAULT_ID_COL = "id"
DEFAULT_PRED_COL = "predicted"
TRUE_LABEL_COL = "_internal_true_label_"
EXTRACTED_LANG_COL = "_extracted_lang_"
POSITIVE_LABEL = 1
FIXED_SEPARATOR = "\t"

# --- Helper Functions ---
def print_stderr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_data(file_path):
    # Minimal load: Assumes clean TSV with header, uses QUOTE_NONE
    # QUOTE_NONE might help if quotes in data were causing parsing issues
    try:
        df = pd.read_csv(file_path, sep=FIXED_SEPARATOR, quoting=csv.QUOTE_NONE)
        return df
    except Exception as e:
        # Keep basic exception print for fatal loading errors
        print_stderr(f"Fatal error loading '{file_path}': {e}")
        return None

def calculate_standard_metrics(y_true, y_pred, pos_label=POSITIVE_LABEL):
    # Assumes valid inputs
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='binary',
        pos_label=pos_label,
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
    extract_lang_from_id,
):
    results = {}

    # --- Load Data ---
    pred_df = load_data(pred_file)
    gold_df = load_data(gold_file)

    if pred_df is None or gold_df is None:
        # Keep minimal check for load failure
        print_stderr("Evaluation failed: Could not load one or both input files.")
        return results

    # --- Prepare & Merge ---
    # Assumes columns exist and merge works
    pred_df_select = pred_df[[id_col, pred_col]]
    gold_df_select = gold_df[[id_col, gold_col]].rename(columns={gold_col: TRUE_LABEL_COL})
    merged_df = pd.merge(gold_df_select, pred_df_select, on=id_col, how="inner")

    if merged_df.empty:
         print_stderr("Warning: Merge resulted in an empty DataFrame. No matching IDs found?")
         # Continue to return empty results if merge fails
         return results

    # --- Extract Labels ---
    # Assumes columns exist and are convertible to int
    y_true_all = merged_df[TRUE_LABEL_COL].astype(int)
    y_pred_all = merged_df[pred_col].astype(int)

    # --- Calculate Overall Metrics ---
    overall_metrics = calculate_standard_metrics(y_true_all, y_pred_all)
    for metric_name, value in overall_metrics.items():
        results[f"multilingual_{metric_name}"] = value

    # --- Calculate Per-Language Metrics ---
    if extract_lang_from_id:
        # Assumes id_col exists and has 'lang_' prefix format
        merged_df[EXTRACTED_LANG_COL] = merged_df[id_col].str.split('_', n=1, expand=True)[0]
        lang_col_to_use = EXTRACTED_LANG_COL

        if lang_col_to_use in merged_df:
            unique_langs = merged_df[lang_col_to_use].dropna().unique()
            for lang in unique_langs:
                lang_str = str(lang)
                lang_mask = merged_df[lang_col_to_use] == lang
                y_true_lang = y_true_all[lang_mask]
                y_pred_lang = y_pred_all[lang_mask]

                # Minimal check if subset is empty after masking
                if not y_true_lang.empty:
                    lang_metrics = calculate_standard_metrics(y_true_lang, y_pred_lang)
                    for metric_name, value in lang_metrics.items():
                        results[f"{lang_str}_{metric_name}"] = value
                # else: # Optionally warn about empty language subsets
                #    print_stderr(f"Warning: No data for language '{lang_str}' after filtering.")

    return results

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate classification results from TSV files (Minimal Version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pred_file", help="Path to prediction TSV file.")
    parser.add_argument("gold_file", help="Path to ground truth TSV file.")
    parser.add_argument("--gold_col", required=True, help="Name of true label column.")
    parser.add_argument("--id_col", default=DEFAULT_ID_COL, help="Name of ID column.")
    parser.add_argument("--pred_col", default=DEFAULT_PRED_COL, help="Name of predicted label column.")
    parser.add_argument("--extract_lang_from_id", action='store_true',
                        help="Extract language from ID prefix.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    results = evaluate(
        pred_file=args.pred_file,
        gold_file=args.gold_file,
        id_col=args.id_col,
        pred_col=args.pred_col,
        gold_col=args.gold_col,
        extract_lang_from_id=args.extract_lang_from_id,
    )

    print(json.dumps(results, sort_keys=True))

if __name__ == "__main__":
    main()
