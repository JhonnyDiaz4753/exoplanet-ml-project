import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)
# === Función de evaluación ===
import torch


def evaluate_model(model, test_set, output_dir='eval_figures/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    test_loader = DataLoader(test_set, batch_size=32)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            all_probs.extend(outputs.cpu().numpy().squeeze())
            all_labels.extend(yb.numpy().squeeze())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    y_pred = (all_probs >= 0.5).astype(int)

    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC Mejorada')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()

    cm = confusion_matrix(all_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Matriz de Confusión Mejorada")
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

    acc = accuracy_score(all_labels, y_pred)
    prec = precision_score(all_labels, y_pred)
    rec = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f'Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n')

    print(f"Evaluación guardada en: {output_dir}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}