import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Dropout, Input
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. Generate Synthetic Stock Price Time Series
# ============================================================

def generate_stock_prices(num_samples=500, seq_length=60):
    """Generate synthetic stock price data with trend and noise"""

    prices = []
    sequences = []

    base_price = 100

    trend = np.random.normal(
        0.001,
        0.01,
        num_samples + seq_length
    )

    price_series = base_price * np.exp(
        np.cumsum(trend)
    )

    for i in range(num_samples):

        sequences.append(
            price_series[i:i + seq_length]
        )

        prices.append(
            price_series[i + seq_length]
        )

    return np.array(sequences), np.array(prices)


# ============================================================
# 2. Generate Data
# ============================================================

X, y = generate_stock_prices(
    num_samples=500,
    seq_length=60
)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ============================================================
# 3. Normalize Data
# ============================================================

scaler = MinMaxScaler()

X_normalized = np.zeros_like(X)

for i in range(len(X)):

    X_normalized[i] = scaler.fit_transform(
        X[i].reshape(-1, 1)
    ).flatten()


y_normalized = scaler.fit_transform(
    y.reshape(-1, 1)
).flatten()


# ============================================================
# 4. Split Data
# ============================================================

split_idx = 400

X_train = X_normalized[:split_idx].reshape(
    split_idx,
    60,
    1
)

X_test = X_normalized[split_idx:].reshape(
    len(X_normalized) - split_idx,
    60,
    1
)

y_train = y_normalized[:split_idx]

y_test = y_normalized[split_idx:]


print(
    f"Training shape: {X_train.shape}, {y_train.shape}"
)

print(
    f"Test shape: {X_test.shape}, {y_test.shape}"
)


# ============================================================
# 5. Build LSTM Model
# ============================================================

model_lstm = Sequential([

    Input(shape=(60, 1)),

    LSTM(
        100,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        50,
        activation="relu",
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        25,
        activation="relu"
    ),

    Dense(1)
])


# ============================================================
# 6. Compile LSTM
# ============================================================

model_lstm.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\n===== LSTM MODEL =====")

model_lstm.summary()


# ============================================================
# 7. Train LSTM
# ============================================================

print("\nTraining LSTM...")

history_lstm = model_lstm.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.2,

    verbose=1
)


# ============================================================
# 8. LSTM Predictions
# ============================================================

y_pred_lstm = model_lstm.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# 9. Build Simple RNN for Comparison
# ============================================================

model_rnn = Sequential([

    Input(shape=(60, 1)),

    SimpleRNN(
        100,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.2),

    SimpleRNN(
        50,
        activation="relu",
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        25,
        activation="relu"
    ),

    Dense(1)
])


# ============================================================
# 10. Compile RNN
# ============================================================

model_rnn.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\n===== SIMPLE RNN MODEL =====")

model_rnn.summary()


# ============================================================
# 11. Train Simple RNN
# ============================================================

print("\nTraining Simple RNN...")

history_rnn = model_rnn.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.2,

    verbose=1
)


# ============================================================
# 12. RNN Predictions
# ============================================================

y_pred_rnn = model_rnn.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# 13. Calculate RMSE
# ============================================================

rmse_lstm = np.sqrt(
    np.mean(
        (y_test - y_pred_lstm) ** 2
    )
)

rmse_rnn = np.sqrt(
    np.mean(
        (y_test - y_pred_rnn) ** 2
    )
)


# ============================================================
# 14. Evaluate Models
# ============================================================

lstm_loss, lstm_mae = model_lstm.evaluate(
    X_test,
    y_test,
    verbose=0
)

rnn_loss, rnn_mae = model_rnn.evaluate(
    X_test,
    y_test,
    verbose=0
)


# ============================================================
# 15. Display Results
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nLSTM:")

print(
    f"Test Loss: {lstm_loss:.6f}"
)

print(
    f"Test MAE: {lstm_mae:.6f}"
)

print(
    f"Test RMSE: {rmse_lstm:.6f}"
)


print("\nSimple RNN:")

print(
    f"Test Loss: {rnn_loss:.6f}"
)

print(
    f"Test MAE: {rnn_mae:.6f}"
)

print(
    f"Test RMSE: {rmse_rnn:.6f}"
)


# ============================================================
# 16. Compare Models
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

if rmse_lstm < rmse_rnn:

    print("LSTM performs better than Simple RNN.")

else:

    print("Simple RNN performs better than LSTM.")


# ============================================================
# 17. Visualization
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8)
)


# ------------------------------------------------------------
# LSTM Training Loss
# ------------------------------------------------------------

axes[0, 0].plot(
    history_lstm.history["loss"],
    label="LSTM Training Loss"
)

axes[0, 0].plot(
    history_lstm.history["val_loss"],
    label="LSTM Validation Loss"
)

axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("LSTM Model Loss")

axes[0, 0].legend()
axes[0, 0].grid()


# ------------------------------------------------------------
# RNN Training Loss
# ------------------------------------------------------------

axes[0, 1].plot(
    history_rnn.history["loss"],
    label="RNN Training Loss"
)

axes[0, 1].plot(
    history_rnn.history["val_loss"],
    label="RNN Validation Loss"
)

axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Simple RNN Model Loss")

axes[0, 1].legend()
axes[0, 1].grid()


# ------------------------------------------------------------
# Predictions Comparison
# ------------------------------------------------------------

axes[1, 0].plot(
    y_test[:100],
    label="Actual"
)

axes[1, 0].plot(
    y_pred_lstm[:100],
    label="LSTM Prediction"
)

axes[1, 0].plot(
    y_pred_rnn[:100],
    label="RNN Prediction"
)

axes[1, 0].set_xlabel("Sample")
axes[1, 0].set_ylabel("Normalized Price")
axes[1, 0].set_title("LSTM vs RNN Predictions")

axes[1, 0].legend()
axes[1, 0].grid()


# ------------------------------------------------------------
# RMSE Comparison
# ------------------------------------------------------------

models = ["LSTM", "Simple RNN"]
rmse_values = [rmse_lstm, rmse_rnn]

axes[1, 1].bar(
    models,
    rmse_values
)

axes[1, 1].set_xlabel("Model")
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].set_title("RMSE Comparison")

axes[1, 1].grid(axis="y")


plt.tight_layout()
plt.show()


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 8 COMPLETED")
print("=" * 60)
