import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLightCurveClassifier(nn.Module):
    def __init__(self):
        super(CNNLightCurveClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(0.5)

        # Usamos AdaptiveAvgPool1d para evitar hardcodear tamaños
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)  # 256 canales x 1
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout_conv(x)

        x = self.global_pool(x).squeeze(-1)  # (B, 256, 1) -> (B, 256)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

def build_model():
    model = CNNLightCurveClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    return model, optimizer, criterion, scheduler
