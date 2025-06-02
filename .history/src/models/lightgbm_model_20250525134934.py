import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
import os

def train_lightgbm_model(csv_path='../../data/features/features_tsfresh.csv', output_dir='../reports/figures/LightGBM/', return_metrics=False):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar el CSV
    df = pd.read_csv(csv_path)
    # Eliminar columnas no numéricas innecesarias como 'Unnamed: 0'
    df = df.loc[:, df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]


    # 2. Separar características y etiquetas
    if 'label' not in df.columns:
        raise ValueError("No se encontró la columna 'label' en el archivo CSV.")
    
    X = df.drop(columns=['label'])
    y = df['label']

    # 3. División de los datos
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

    # 4. Modelo LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss'
    )

    # 5. Evaluación
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Guardar métricas
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC LightGBM')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Matriz de Confusión LightGBM")
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

    print(f"Modelo LightGBM entrenado. Resultados guardados en {output_dir}")
  
    if return_metrics:
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }

 
