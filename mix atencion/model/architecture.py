# model/architecture.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
import numpy as np


class Attention(layers.Layer):
    """Mecanismo de Atencion simple"""

    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        e = tf.matmul(x, self.W) + self.b
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context


def create_lstm_attention_model(input_shape, num_classes, config):
    """Crea modelo LSTM + Atencion"""
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)

    x = Attention()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def create_cnn_lstm_model(input_shape, num_classes, config):
    """Crea modelo CNN + LSTM (alternativa)"""
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3)
    )(x)

    x = Attention()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def create_svm_model(X_train, y_train):
    """Modelo SVM para comparacion (sin secuencias)"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X_flat = X_train.reshape(X_train.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_scaled, y_train)

    return svm, scaler
