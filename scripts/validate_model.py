import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt

dropout_rate = 0.2
l2_lambda = 0.001

# Laden des Datensatzes
df = pd.read_csv('data\\TestData\\testDataSetComplete.csv')


class MSELogger(Callback):
    def __init__(self):
        super(MSELogger, self).__init__()
        self.mse_per_sample = []

    def on_train_batch_end(self, batch, logs=None):
        mse = logs.get('loss')
        self.mse_per_sample.append(mse)


mse_logger = MSELogger()


# Annahme: Die letzte Spalte ist das Ziel (y), die anderen sind Features (X)

# Normalisieren der Daten
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


len_pred = 10
X_sequences = create_sequences(X_scaled, len_pred)


def init_autoencoder(num_features):
    global autoencoder
    inputs = Input(shape=(len_pred, num_features))
    encoded = LSTM(num_features*4, activation='relu',
                   kernel_regularizer=l2(l2_lambda))(inputs)
    encoded = Dropout(dropout_rate)(encoded)

    decoded = RepeatVector(len_pred)(encoded)
    decoded = LSTM(num_features*4, activation='relu',
                   return_sequences=True,
                   kernel_regularizer=l2(l2_lambda))(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    outputs = TimeDistributed(Dense(num_features))(decoded)
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
# Kompilieren des Modells


model = init_autoencoder(7)
model.compile(optimizer='adam', loss='mse')

# Trainieren des Modells und Speichern der Trainingshistorie
history = model.fit(X_sequences, X_sequences, epochs=1, batch_size=32, callbacks=[mse_logger])
mse_df = pd.DataFrame(mse_logger.mse_per_sample, columns=['MSE'])
mse_df.to_csv('mse_per_sample.csv', index=False)

# Plotten der MSE-Werte
plt.plot(mse_logger.mse_per_sample)
plt.xlabel('Sample')
plt.ylabel('MSE')
plt.title('MSE nach jedem Sample')
plt.show()
