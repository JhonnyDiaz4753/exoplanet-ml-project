import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, dropout_p=0.3):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_p)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += identity
        return F.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.layer1 = BasicBlock1D(1, 16, dropout_p=dropout_p)
        self.layer2 = BasicBlock1D(16, 32, downsample=True, dropout_p=dropout_p)
        self.layer3 = BasicBlock1D(32, 64, downsample=True, dropout_p=dropout_p)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x)

def add_noise(x, noise_level=0.02, p=0.5):
    # Añade ruido gaussiano con probabilidad p al batch
    if np.random.rand() < p:
        noise = noise_level * torch.randn_like(x)
        return x + noise
    return x

def train_model(X, y, output_path='models/cnn_model.pt', epochs=100, batch_size=32, lr=0.001, patience=10):
    print("Training improved CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D().to(device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
               'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for xb, yb in train_loader: 
            xb, yb = xb.to(device), yb.to(device)

            # Data augmentation: añadir ruido
            xb = add_noise(xb, noise_level=0.02, p=0.5)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

            all_preds.extend((pred >= 0.5).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds)
        train_rec = recall_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.extend((pred >= 0.5).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_prec'].append(train_prec)
        history['val_prec'].append(val_prec)
        history['train_rec'].append(train_rec)
        history['val_rec'].append(val_rec)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return model, history, test_set

def evaluate_model(model, test_set, output_dir='../reports/figures/EvaluacionCNN_Improved/'):
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
        f.write('Curva ROC para el modelo CNN Mejorado\n')
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
        f.write('Matriz de confusión del modelo CNN Mejorado\n')
        f.write(str(cm))
        f.write('\nFilas = clases reales, Columnas = predichas\n')

    # Métricas finales
    acc = accuracy_score(all_labels, y_pred)
    prec = precision_score(all_labels, y_pred)
    rec = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write('Métricas finales del modelo CNN Mejorado\n')
        f.write(f'Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n')

    print(f"Evaluación guardada en: {output_dir}")
