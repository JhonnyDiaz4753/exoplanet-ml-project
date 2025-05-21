# src/data/preprocess.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SEGMENT_LENGTH = 2000  # número de puntos de tiempo a conservar

os.makedirs(PROCESSED_DIR + "/positive", exist_ok=True)
os.makedirs(PROCESSED_DIR + "/negative", exist_ok=True)

def clean_curve(df):
    """Limpia una curva de luz: elimina NaNs, normaliza y recorta/pad."""
    df = df.dropna()
    
    if len(df) < SEGMENT_LENGTH:
        return None  # descartamos curvas demasiado cortas
    
    df = df.iloc[:SEGMENT_LENGTH].copy()
    
    # Normalización robusta
    flux = df["flux"].values
    median = np.median(flux)
    std = np.std(flux)
    df["flux"] = (flux - median) / std
    
    return df[["time", "flux"]]

def process_all_curves():
    for label in ["positive", "negative"]:
        in_dir = os.path.join(RAW_DIR, label)
        out_dir = os.path.join(PROCESSED_DIR, label)
        files = [f for f in os.listdir(in_dir) if f.endswith(".csv")]

        print(f"Procesando {len(files)} curvas de clase '{label}'...")

        for file in tqdm(files):
            path = os.path.join(in_dir, file)
            df = pd.read_csv(path)

            clean_df = clean_curve(df)
            if clean_df is not None:
                clean_df.to_csv(os.path.join(out_dir, file), index=False)

        print(f"✅ Clase '{label}' lista.")

if __name__ == "__main__":
    process_all_curves()
