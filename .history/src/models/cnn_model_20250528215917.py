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
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN1D(nn.Module):
    def __init__(self, dropout=0.3):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Calcular flatten_dim dinámicamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 200)  # (batch, channels, length)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


# === Añadir ruido a datos para Data Augmentation ===
def add_noise(X, noise_level=0.02):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)

# === Bloque ResNet 1D ===
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
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

# === Self-Attention Manual ===
class ManualSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, V)
        return context.mean(dim=1)

# === Modelo CNN + LSTM + Attention mejorado ===
class ResNetLSTM1DWithManualAttention(nn.Module):
    def __init__(self, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.layer1 = BasicBlock1D(1, 16)
        self.layer2 = BasicBlock1D(16, 32, downsample=True)
        self.layer3 = BasicBlock1D(32, 64, downsample=True)

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True)

        self.attn = ManualSelfAttention(embed_dim=lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.attn(lstm_out)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# === Función de entrenamiento mejorada ===
def train_model(X, y, output_path='cnn_lstm_attn_model.pt', epochs=50, batch_size=128, lr=1e-4,
                patience=10, model_name='', augment_noise=False,  weight_decay=1e-5,
    lstm_hidden=128, lstm_layers=2, dropout=0.3):

    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")

    # Augmentación
    if augment_noise:
        X = np.concatenate([X, add_noise(X)])
        y = np.concatenate([y, y])

    # Modelo (reducido para mejorar velocidad)
    if model_name == 'cnn_lstm_attn':
        model = ResNetLSTM1DWithManualAttention(
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout
        ).to(device)
        print("Training CNN + LSTM + Attention model (optimiz.) with augmentation =", augment_noise)
    else:
        print("Training Simple CNN model (optimiz.) with augmentation =", augment_noise)
        model = SimpleCNN1D(dropout=dropout).to(device)
       

    # Preparar dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    # DataLoader optimizados
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=4, pin_memory=True)

    # Entrenamiento
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            print("Shape de xb antes de model:", xb.shape)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            correct += ((pred > 0.5) == yb).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
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

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
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



# === Función de evaluación ===
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