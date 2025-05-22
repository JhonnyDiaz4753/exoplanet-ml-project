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
    valid_labels = []  # Solo etiquetas de curvas válidas

    for i, df in enumerate(dataframes):
        df = df.dropna()
        if df["flux"].isnull().any() or df.shape[0] < 10:
            continue  # Descartamos si aún quedan NaNs o es muy corta

        df_ = df.copy()
        df_["id"] = i
        all_data.append(df_[["time", "flux"]])
        valid_labels.append(labels[i])

    if not all_data:
        raise ValueError("No hay datos válidos para extraer características.")
        
    all_data = pd.concat(all_data)
    
    # Extraer características
    extracted_features = extract_features( column_sort="time", disable_progressbar=False)

    # Imputar valores faltantes (en features, no en curvas)
    impute(extracted_features)

    # Seleccionar características relevantes
    y = pd.Series(valid_labels)
    features_filtered = select_features(extracted_features, y)

    return features_filtered, y
