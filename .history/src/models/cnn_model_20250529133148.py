import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization

def build_model():
    model = Sequential([
        # Primera capa convolucional
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(1999, 2)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),

        # Segunda capa convolucional
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),

        # Tercera capa convolucional
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),

        # Capa densa final
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
