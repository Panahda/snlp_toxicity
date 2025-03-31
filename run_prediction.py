import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from detoxify import Detoxify

PREDICTION_COL = 'toxicity'

def load_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, quoting=3)
    return df

def batch_df(input_df, batch_size=32):
    total_rows = len(input_df)
    for start_index in range(0, total_rows, batch_size):
        end_index = min(start_index + batch_size, total_rows)
        yield input_df.iloc[start_index:end_index]

def process_res(results_dict, index_ids, prediction_col):
    """
    Processes the results dictionary into a final DataFrame with 'id' and 'predicted' columns.
    """
    res_df = pd.DataFrame(
    results_dict,  
    index=index_ids, 
    ).round(5)
    
    res_df['predicted'] = (res_df[prediction_col] >= 0.5).astype(int)
    final_df = res_df.reset_index()
    final_df = final_df.rename(columns={'index': 'id'})
    final_df = final_df[['id', 'predicted']]
    return final_df

def run(model_name, input_df, id_col, text_col, prediction_col, from_ckpt, device="cpu", batch_size=32):
    """Loads model and runs inference on the input_obj in batches."""
    
    if model_name is not None:
        print(f"Loading model: {model_name} on device: {device}")
        model = Detoxify(model_name, device=device)
    elif from_ckpt is not None:
        print(f"Loading model from checkpoint: {from_ckpt} on device: {device}")
        model = Detoxify(checkpoint=from_ckpt, device=device)

    results_dict = {}
    index_ids = []
    first_batch = True
    processed_batches = 0

    try:
        for n, batch_data in enumerate(batch_df(input_df, batch_size=batch_size)):
            batch_ids = batch_data[id_col].tolist()
            batch_texts = batch_data[text_col].tolist()
            print(f"batch = {n} (size: {len(batch_texts)})")
            
            res = model.predict(batch_texts) 

            # On the first batch, initialize the results_dict keys
            if first_batch:
                for key in res.keys():
                    results_dict[key] = []
                first_batch = False

            # Append scores for this batch to the corresponding lists
            for key, scores in res.items():
                if key in results_dict: # Ensure key exists before extending
                     results_dict[key].extend(scores)
                else:
                    print(f"Warning: Encountered unexpected score key '{key}' in batch {n}.")
                    results_dict[key] = scores 
            
            index_ids.extend(batch_ids)
            processed_batches += 1
            
        res_df = process_res(results_dict, index_ids, prediction_col)    
        return res_df

    except FileNotFoundError as e:
         print(f"Error: {e}")
    except ValueError as e:
         print(f"Error: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")

                
def export_tsv(res_df, dest_file):
    if dest_file is not None:
        dest_dir = os.path.dirname(dest_file)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        res_df.to_csv(dest_file, sep='\t', index=False)
        print("Results saved successfully.")
    else:
        print("Destination file not specified. Results not saved.")

if __name__ == "__main__":
    input_file_type = "dev"
    input_file = f"data/{input_file_type}.tsv"
    text_col = "text"
    id_col = "id"
    model = "multilingual" # or "original", "unbiased"
    from_ckpt_path = None
    device_type = "cpu"
    batch_processing_size = 32
    dest_file = f"res/result_{input_file_type}.tsv"
    
    input_df = load_tsv(input_file)
    
    print("--- Starting Prediction ---")
    res_df = run(
        model_name=model,
        input_df=input_df,
        id_col=id_col,
        text_col=text_col,
        prediction_col=PREDICTION_COL,
        from_ckpt=from_ckpt_path,
        device=device_type,
        batch_size=batch_processing_size
    )
    print("--- Prediction Finished and Exporting to TSV ---")
    export_tsv(res_df, dest_file)
    print("--- TSV exported ---")
        