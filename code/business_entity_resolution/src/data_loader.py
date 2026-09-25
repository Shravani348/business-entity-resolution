import os
import pandas as pd

def load_data_from_dir(directory):
    """
    Loads all TSV files from a given directory into a dictionary of DataFrames.
    Keys are the filenames (without .tsv).
    """
    data = {}
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' not found.")
        return data

    for file in os.listdir(directory):
        if file.endswith('.tsv') and not file.startswith('._'):
            name = os.path.splitext(file)[0]
            parquet_path = os.path.join(directory, f"{name}.parquet")
            
            if os.path.exists(parquet_path):
                print(f"Loading cached {name}.parquet...", flush=True)
                df = pd.read_parquet(parquet_path)
            else:
                filepath = os.path.join(directory, file)
                print(f"Reading {file} and caching to parquet...", flush=True)
                df = pd.read_csv(filepath, sep='\t', dtype=str)
                df.to_parquet(parquet_path, engine='pyarrow')
                
            data[name] = df
            
    return data
