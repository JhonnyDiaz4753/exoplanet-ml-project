# src/data/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
SEGMENT_LENGTH = 2000
SAVGOL_WINDOW = 101  # Debe ser impar y < SEGMENT_LENGTH

def clean_curve(df, star_name):
    if "pdcsap_flux" not in df.columns or "timecorr" not in df.columns:
        print(f"❌ {star_name}: faltan columnas necesarias")
        return None

    df = df[["timecorr", "pdcsap_flux"]].rename(columns={"timecorr": "time", "pdcsap_flux": "flux"})
    df = df.dropna()
    if len(df) < 100:
        print(f"❌ {star_name}: muy corta ({len(df)} puntos)")
        return None

    # Interpolación para alcanzar SEGMENT_LENGTH
    try:
        time = df["time"].values
        flux = df["flux"].values

        if len(df) < SEGMENT_LENGTH:
            new_time = np.linspace(time[0], time[-1], SEGMENT_LENGTH)
            new_flux = np.interp(new_time, time, flux)
        else:
            new_time = time[:SEGMENT_LENGTH]
            new_flux = flux[:SEGMENT_LENGTH]

        # Suavizado
        if len(new_flux) >= SAVGOL_WINDOW:
            new_flux = savgol_filter(new_flux, window_length=SAVGOL_WINDOW, polyorder=3)

        # Normalización robusta
        median = np.median(new_flux)
        std = np.std(new_flux) if np.std(new_flux) > 0 else 1
        norm_flux = (new_flux - median) / std

        return pd.DataFrame({"time": new_time, "flux": norm_flux})

    except Exception as e:
        print(f"❌ {star_name}: error en limpieza - {e}")
        return None

def save_split_dataset(curves, labels, label_name):
    train_dir = PROCESSED_DIR / "train" / label_name
    val_dir   = PROCESSED_DIR / "val" / label_name
    test_dir  = PROCESSED_DIR / "test" / label_name

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    X_trainval, X_test = train_test_split(curves, test_size=150, random_state=42)
    X_train, X_val = train_test_split(X_trainval, test_size=150, random_state=42)

    for i, df in enumerate(X_train):
        df.to_csv(train_dir / f"{label_name}_{i}.csv", index=False)
    for i, df in enumerate(X_val):
        df.to_csv(val_dir / f"{label_name}_{i}.csv", index=False)
    for i, df in enumerate(X_test):
        df.to_csv(test_dir / f"{label_name}_{i}.csv", index=False)

def process_all_curves():
    for label in ["positive", "negative"]:
        in_dir = RAW_DIR / label
        files = list(in_dir.glob("*.csv"))

        print(f"\n🔍 Procesando {len(files)} curvas de clase '{label}'...")
        cleaned = []

        for file_path in tqdm(files):
            try:
                df = pd.read_csv(file_path)
                clean_df = clean_curve(df, file_path.name)
                if clean_df is not None:
                    cleaned.append(clean_df)
                else:
                    print(f"⚠️ {file_path.name} fue descartada.")
            except Exception as e:
                print(f"❌ Error procesando {file_path.name}: {e}")

        if len(cleaned) < 1000:
            print(f"❌ No hay suficientes curvas válidas en '{label}' (tienes {len(cleaned)})")
            continue

        save_split_dataset(cleaned[:1000], [label]*1000, label)
        print(f"✅ Clase '{label}' guardada (1000 curvas divididas en train/val/test)")

if __name__ == "__main__":
    # Limpiar directorio anterior
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    process_all_curves()
