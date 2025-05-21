# src/data/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 📌 Ruta base segura relativa al archivo actual
BASE_DIR = Path(__file__).resolve().parents[2]  # va 2 niveles arriba
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
SEGMENT_LENGTH = 2000

# Crear directorios
(PROCESSED_DIR / "positive").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "negative").mkdir(parents=True, exist_ok=True)

def clean_curve(df, star_name):
    df = df.dropna()

    if len(df) < 100:
        print(f"❌ {star_name}: muy corta ({len(df)} puntos)")
        return None

    if len(df) >= SEGMENT_LENGTH:
        df = df.iloc[:SEGMENT_LENGTH].copy()
    else:
        padding = pd.DataFrame({
            "time": [np.nan] * (SEGMENT_LENGTH - len(df)),
            "flux": [np.nan] * (SEGMENT_LENGTH - len(df))
        })
        df = pd.concat([df, padding], ignore_index=True)

    flux = df["flux"].values
    median = np.nanmedian(flux)
    std = np.nanstd(flux) if np.nanstd(flux) > 0 else 1
    df["flux"] = (flux - median) / std

    return df[["time", "flux"]]

def process_all_curves():
    for label in ["positive", "negative"]:
        in_dir = RAW_DIR / label
        out_dir = PROCESSED_DIR / label
        files = list(in_dir.glob("*.csv"))

        print(f"\n🔍 Procesando {len(files)} curvas de clase '{label}'...")

        for file_path in tqdm(files):
            try:
                df = pd.read_csv(file_path)
                clean_df = clean_curve(df, file_path.name)
                if clean_df is not None:
                    clean_df.to_csv(out_dir / file_path.name, index=False)
                else:
                    print(f"⚠️ {file_path.name} fue descartada.")
            except Exception as e:
                print(f"❌ Error procesando {file_path.name}: {e}")

        print(f"✅ Clase '{label}' terminada.")

if __name__ == "__main__":
    process_all_curves()
