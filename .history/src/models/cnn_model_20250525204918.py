import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# === SEED PARA REPRODUCIBILIDAD ===================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === BLOQUE RESNET 1D ============================================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


# === MODELO CNN + LSTM + ATTENTION ================================
class ResNetLSTM1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.resnet = nn.Sequential(
            BasicBlock1D(in_channels, 64),
            BasicBlock1D(64, 64),
            BasicBlock1D(64, 128, downsample=True),
            BasicBlock1D(128, 128),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)  # (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, length, channels)
        lstm_out, _ = self.lstm(x)  # (batch, length, 2*hidden)

        attn_weights = self.attention(lstm_out)  # (batch, length, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, 2*hidden)

        out = self.classifier(context)  # (batch, num_classes)
        return out


# === EVALUACIÓN DEL MODELO ========================================
def evaluate_model(model, val_loader, device, output_dir):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Curva ROC y AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Modelo Mejorado CNN + LSTM + Attention')
    plt.legend(loc='lower right')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()

    with open(f'{output_dir}/roc_curve.txt', 'w') as f:
        f.write('Curva ROC del modelo mejorado CNN + LSTM + Attention.\n')
        f.write(f'AUC: {roc_auc:.4f}\n')
        f.write('FPR: False Positive Rate\n')
        f.write('TPR: True Positive Rate\n')

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Modelo Mejorado CNN + LSTM + Attention')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

    with open(f'{output_dir}/confusion_matrix.txt', 'w') as f:
        f.write('Matriz de Confusión del modelo mejorado\n')
        f.write(f'{cm}\n')
        f.write('Filas: valores reales; Columnas: predicciones del modelo.\n')

    # Métricas adicionales
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write('Métricas de evaluación del modelo mejorado:\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall: {rec:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')


def train_model(X, y, output_path='../models/cnn_model.pt', epochs=100, batch_size=16, lr=1e-4, patience=10, model_name='cnn_lstm_attn'):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'cnn_lstm':
     model = ResNetLSTM1D().to(device)
     print("Training Improved CNN + LSTM model...")
    elif model_name == 'cnn_lstm_attn':
     print("Training Improved CNN + LSTM+ Attention model...")
     model = ResNetLSTM1DWithManualAttention().to(device)
    elif model_name == 'transformer':
     model = Transformer1D().to(device)
     print("Training Improved Transformer model...")
    else:
     raise ValueError(f"Modelo no reconocido: {model_name}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            correct += ((pred > 0.5) == yb).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += ((pred > 0.5) == yb).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return model, history, test_set


def evaluate_model(model, test_set, output_dir='reports/figures/EvaluacionCNN_Improved/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    test_loader = DataLoader(test_set, batch_size=32)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            probs = outputs.cpu().numpy().squeeze()
            labels = yb.numpy().squeeze()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    y_pred = (all_probs >= 0.5).astype(int)

    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Curva ROC
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

    with open(f'{output_dir}/roc_curve.txt', 'w') as f:
        f.write('Curva ROC para el modelo CNN + LSTM Mejorado\n')
        f.write(f'AUC: {auc:.4f}\n')
        f.write('Muestra la tasa de verdaderos positivos frente a la de falsos positivos.\n')

    # Matriz de Confusión
    cm = confusion_matrix(all_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Matriz de Confusión Mejorada")
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

    with open(f'{output_dir}/confusion_matrix.txt', 'w') as f:
        f.write('Matriz de confusión del modelo CNN + LSTM Mejorado\n')
        f.write(str(cm))
        f.write('\nFilas = clases reales, Columnas = predichas\n')

    # Métricas finales
    acc = accuracy_score(all_labels, y_pred)
    prec = precision_score(all_labels, y_pred)
    rec = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write('Métricas finales del modelo CNN + LSTM Mejorado\n')
        f.write(f'Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n')

    print(f"Evaluación guardada en: {output_dir}")
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }
