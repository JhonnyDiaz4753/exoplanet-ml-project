# src/data/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
import shutil
import random

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

SEGMENT_LENGTH = 1624
MIN_LENGTH = 1624  # para evitar padding plano
SAVGOL_WINDOW = 101  # impar y < SEGMENT_LENGTH
SAVGOL_POLYORDER = 3

# Crear estructura de carpetas
for split in ["train", "val", "test"]:
    for label in ["positive", "negative"]:
        (PROCESSED_DIR / split / label).mkdir(parents=True, exist_ok=True)

def clean_curve(df, star_name):
    if "pdcsap_flux" not in df.columns or "timecorr" not in df.columns:
        print(f"❌ {star_name}: faltan columnas necesarias")
        return None

    df = df[["timecorr", "pdcsap_flux"]].rename(columns={"timecorr": "time", "pdcsap_flux": "flux"})
    df = df.dropna()
    
    if len(df) < MIN_LENGTH:
        print(f"❌ {star_name}: muy corta ({len(df)} puntos)")
        return None

    df = df.iloc[:SEGMENT_LENGTH].copy()

    # Suavizado con Savitzky-Golay
    try:
        flux_smoothed = savgol_filter(df["flux"].values, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)
    except Exception as e:
        print(f"⚠️ {star_name}: error en suavizado ({e})")
        return None

    # Normalización robusta
    median = np.median(flux_smoothed)
    std = np.std(flux_smoothed) if np.std(flux_smoothed) > 0 else 1
    df["flux"] = (flux_smoothed - median) / std

    return df[["time", "flux"]]

def process_all_curves():
    for label in ["positive", "negative"]:
        in_dir = RAW_DIR / label
        files = list(in_dir.glob("*.csv"))
        cleaned = []

        print(f"\n🔍 Procesando {len(files)} curvas de clase '{label}'...")

        for file_path in tqdm(files):
            try:
                df = pd.read_csv(file_path)
                clean_df = clean_curve(df, file_path.name)
                if clean_df is not None:
                    cleaned.append((file_path.name, clean_df))
                else:
                    print(f"⚠️ {file_path.name} fue descartada.")
            except Exception as e:
                print(f"❌ Error procesando {file_path.name}: {e}")

        # Shuffle y división 700/150/150
        if len(cleaned) < 1000:
            print(f"❌ No hay suficientes curvas válidas en '{label}' (tienes {len(cleaned)})")
            continue

        random.seed(42)
        random.shuffle(cleaned)

        split_counts = {"train": 700, "val": 150, "test": 150}
        idx = 0

        for split, count in split_counts.items():
            for i in range(count):
                name, df = cleaned[idx]
                out_path = PROCESSED_DIR / split / label / name
                df.to_csv(out_path, index=False)
                idx += 1

        print(f"✅ Clase '{label}' dividida y guardada: 700/150/150.")

if __name__ == "__main__":
    process_all_curves()
