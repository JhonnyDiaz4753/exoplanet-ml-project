import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
REPORT_DIR = "reports/figures"
SEGMENT_LENGTH = 2000  # puntos por curva

os.makedirs(PROCESSED_DIR + "/positive", exist_ok=True)
os.makedirs(PROCESSED_DIR + "/negative", exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def clean_curve(df):
    df = df.dropna()
    if len(df) < SEGMENT_LENGTH:
        return None
    df = df.iloc[:SEGMENT_LENGTH].copy()
    flux = df["flux"].values
    median = np.median(flux)
    std = np.std(flux)
    df["flux"] = (flux - median) / std
    return df[["time", "flux"]]

def save_plot_and_description(df, label, name):
    fig_dir = os.path.join(REPORT_DIR, f"{name}_lc")
    os.makedirs(fig_dir, exist_ok=True)

    # Guardar imagen
    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["flux"], color="blue")
    plt.title(f"Curva de luz - {name} ({label})")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Flujo normalizado")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{name}_lc.png"))
    plt.close()

    # Guardar descripción
    description = f"""Curva de luz de la estrella {name}.
Clase: {label.upper()}
Puntos de tiempo: {len(df)}
Longitud normalizada: {SEGMENT_LENGTH}
Ejes: tiempo (en días) vs flujo normalizado (mediana=0, std=1)

Esta curva representa la variación de luminosidad medida por la misión Kepler.
"""
    with open(os.path.join(fig_dir, "description.txt"), "w") as f:
        f.write(description)

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
                name = file.replace("_lc.csv", "").lower()
                clean_df.to_csv(os.path.join(out_dir, file), index=False)
                save_plot_and_description(clean_df, label, name)

        print(f"✅ Clase '{label}' lista.")

if __name__ == "__main__":
    process_all_curves()
