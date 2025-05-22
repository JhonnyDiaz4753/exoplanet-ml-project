# src/features/tsfresh_utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def save_figure_with_description(fig, figure_name, description, base_dir="reports/figures"):
    """
    Guarda la figura con un nombre y genera un txt con su descripción,
    siguiendo la estructura:
    
    reports/figures/NombreDelGrafico/
        ├── image.png
        └── description.txt
    """
    folder_path = os.path.join(base_dir, figure_name.replace(" ", "_"))
    os.makedirs(folder_path, exist_ok=True)
    
    image_path = os.path.join(folder_path, "image.png")
    description_path = os.path.join(folder_path, "description.txt")
    
    fig.savefig(image_path, bbox_inches='tight')
    plt.close(fig)  # Cierra la figura para liberar memoria
    
    with open(description_path, "w") as f:
        f.write(description)

def extract_and_select_features(dataframes, labels):
    """
    dataframes: lista de pd.DataFrame con columnas 'time', 'flux'
    labels: lista con etiqueta para cada dataframe (0 o 1)

    Devuelve: DataFrame con características seleccionadas y vector y con las etiquetas
    """
    all_data = []
    valid_labels = []

    for i, df in enumerate(dataframes):
        df = df.dropna(subset=["time", "flux"])
        if "flux" not in df.columns or "time" not in df.columns:
            print(f"Advertencia: DataFrame {i} no tiene columnas 'time' y 'flux'.")
            continue
        if df["flux"].isnull().any() or df.shape[0] < 10:
            print(f"Advertencia: DataFrame {i} tiene menos de 10 filas o valores nulos en 'flux'.")
            continue

        df_ = df.copy()
        df_["id"] = i
        all_data.append(df_[["id", "time", "flux"]])
        valid_labels.append(labels[i])

    if not all_data:
        raise ValueError("No hay datos válidos para extraer características.")

    all_data = pd.concat(all_data, ignore_index=True)

    # Extraer características
    extracted_features = extract_features(
        all_data,
        column_id="id",
        column_sort="time",
        column_value="flux",
        disable_progressbar=False
    )
    print("Características extraídas:", extracted_features.shape)

    # Imputar valores faltantes
    impute(extracted_features)

    # Seleccionar características relevantes
    y = pd.Series(valid_labels, index=extracted_features.index)
    features_filtered = select_features(extracted_features, y)

    print("Características seleccionadas después del filtrado:", features_filtered.shape)
    if features_filtered.shape[1] == 0:
        print("Advertencia: No se seleccionaron características relevantes. Revisa los datos y las etiquetas.")

    return features_filtered, y