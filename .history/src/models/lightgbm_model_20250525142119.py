import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import joblib

# Limpiar nombres para ser compatibles con LightGBM
def clean_column_name(name):
    name = str(name)
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def save_plot(fig, name, explanation, output_dir):
    fig_dir = os.path.join(output_dir, name)
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{name}.png")
    txt_path = os.path.join(fig_dir, f"{name}.txt")
    fig.savefig(fig_path, bbox_inches='tight')
    with open(txt_path, 'w') as f:
        f.write(explanation)
    plt.close(fig)

def plot_and_save_all_metrics(y_test, y_pred, y_pred_proba, output_dir):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label='ROC Curve')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve - LightGBM')
    ax_roc.legend()
    save_plot(fig_roc, "roc_curve", "Curva ROC que muestra la relación entre tasa de verdaderos positivos (TPR) y tasa de falsos positivos (FPR) para el modelo LightGBM.", output_dir)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label='Precision-Recall Curve')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve - LightGBM')
    ax_pr.legend()
    save_plot(fig_pr, "precision_recall_curve", "Curva Precision-Recall que evalúa el equilibrio entre precisión y exhaustividad del modelo.", output_dir)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix - LightGBM')
    save_plot(fig_cm, "confusion_matrix", "Matriz de confusión que muestra los valores verdaderos frente a las predicciones del modelo LightGBM.", output_dir)

def train_lightgbm_model(csv_path='../../data/features/features_tsfresh.csv',
                         output_dir='../reports/figures/LightGBM/',
                         return_metrics=False):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar CSV
    df = pd.read_csv(csv_path)

    # 2. Columnas numéricas y eliminación de duplicadas
    df = df.loc[:, df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Validar columna label
    if 'label' not in df.columns:
        raise ValueError("No se encontró la columna 'label' en el archivo CSV.")

    # 4. Separar características y etiquetas
    X = df.drop(columns=['label'])
    y = df['label']

   # 5. Limpiar nombres de columnas
    clean_names = [clean_column_name(col) for col in X.columns]
    X.columns = clean_names

# 5.1 Eliminar columnas duplicadas por nombre (mantener la primera aparición)
    X = X.loc[:, ~X.columns.duplicated()]

    # 6. Guardar nombres
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for col in X.columns:
            f.write(col + '\n')

    # 7. División de datos
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

    # 8. Entrenar modelo
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
        eval_metric='binary_logloss',
        early_stopping_rounds=20,
        verbose=False
    )

    # 9. Evaluación
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. Guardar métricas
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    # 11. Guardar gráficas con explicación
    plot_and_save_all_metrics(y_test, y_pred, y_pred_proba, output_dir)

    # 12. Guardar modelo (opcional)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lightgbm_model.pkl")

    print(f"✅ Modelo LightGBM entrenado. Resultados guardados en {output_dir}")
  
    if return_metrics:
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }
