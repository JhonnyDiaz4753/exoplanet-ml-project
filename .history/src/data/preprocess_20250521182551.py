# src/data/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
SEGMENT_LENGTH = 2000

(PROCESSED_DIR / "positive").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "negative").mkdir(parents=True, exist_ok=True)

def clean_curve(df, star_name):
    # Validar que existan las columnas que necesitamos
    if "pdcsap_flux" not in df.columns or "timecorr" not in df.columns:
        print(f"❌ {star_name}: faltan columnas necesarias")
        return None

    df = df[["timecorr", "pdcsap_flux"]].rename(columns={
        "timecorr": "time",
        "pdcsap_flux": "flux"
    })

    df = df.dropna()
    if len(df) < 100:
        print(f"❌ {star_name}: muy corta ({len(df)} puntos)")
        return None

    # Recortar o rellenar hasta SEGMENT_LENGTH
    if len(df) >= SEGMENT_LENGTH:
        df = df.iloc[:SEGMENT_LENGTH].copy()
    else:
        last_time = df["time"].iloc[-1]
        last_flux = df["flux"].iloc[-1]
        padding = pd.DataFrame({
            "time": np.linspace(last_time, last_time + 1, SEGMENT_LENGTH - len(df)),
            "flux": [last_flux] * (SEGMENT_LENGTH - len(df))
        })
        df = pd.concat([df, padding], ignore_index=True)

    # Normalización robusta
    flux = df["flux"].values
    median = np.median(flux)
    std = np.std(flux) if np.std(flux) > 0 else 1
    df["flux"] = (flux - median) / std

    df = df.dropna()  # Última barrera de seguridad

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
