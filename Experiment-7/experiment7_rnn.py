import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dense, Dropout
from keras.optimizers import Adam


# ============================================================
# 1. Generate Synthetic Time Series Data
# ============================================================

def generate_time_series(num_samples=1000, seq_length=50):
    """Generate sine wave time series"""

    data = []
    targets = []

    for _ in range(num_samples):

        start = np.random.uniform(0, 2 * np.pi)

        x = np.sin(
            np.linspace(
                start,
                start + 10 * np.pi,
                seq_length
            )
        )

        data.append(x)

        # Predict last value
        targets.append(x[-1])

    return np.array(data), np.array(targets)


# ============================================================
# 2. Generate Data
# ============================================================

X, y = generate_time_series(
    num_samples=1000,
    seq_length=50
)

# Reshape:
# samples, timesteps, features

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print("Input shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 3. Split Train and Test Data
# ============================================================

split_idx = 800

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 4. Build Simple RNN
# ============================================================

model = Sequential([

    Input(shape=(50, 1)),

    SimpleRNN(
        64,
        activation="relu",
        return_sequences=True
    ),

    SimpleRNN(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


# ============================================================
# 5. Compile Model
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


# ============================================================
# 6. Display Model Architecture
# ============================================================

model.summary()


# ============================================================
# 7. Train Model
# ============================================================

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.2,

    verbose=1
)


# ============================================================
# 8. Evaluate Model
# ============================================================

train_loss = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

test_loss = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\n===== RESULTS =====")

print(
    f"Training Loss: {train_loss[0]:.6f}"
)

print(
    f"Training MAE: {train_loss[1]:.6f}"
)

print(
    f"Test Loss: {test_loss[0]:.6f}"
)

print(
    f"Test MAE: {test_loss[1]:.6f}"
)


# ============================================================
# 9. Make Predictions
# ============================================================

y_pred = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 10. Calculate RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (y_test - y_pred.flatten()) ** 2
    )
)

print(
    f"Test RMSE: {rmse:.6f}"
)


# ============================================================
# 11. Visualization
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8)
)


# ------------------------------------------------------------
# Plot 1: Training History
# ------------------------------------------------------------

axes[0, 0].plot(
    history.history["loss"],
    label="Training Loss"
)

axes[0, 0].plot(
    history.history["val_loss"],
    label="Validation Loss"
)

axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Model Loss")

axes[0, 0].legend()
axes[0, 0].grid()


# ------------------------------------------------------------
# Plot 2: MAE
# ------------------------------------------------------------

axes[0, 1].plot(
    history.history["mae"],
    label="Training MAE"
)

axes[0, 1].plot(
    history.history["val_mae"],
    label="Validation MAE"
)

axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].set_title("Mean Absolute Error")

axes[0, 1].legend()
axes[0, 1].grid()


# ------------------------------------------------------------
# Plot 3: Predictions vs Actual
# ------------------------------------------------------------

axes[1, 0].plot(
    y_test[:100],
    label="Actual",
    marker="o",
    markersize=3
)

axes[1, 0].plot(
    y_pred[:100],
    label="Predicted",
    marker="x",
    markersize=3
)

axes[1, 0].set_xlabel("Sample")
axes[1, 0].set_ylabel("Value")
axes[1, 0].set_title("Predictions vs Actual")

axes[1, 0].legend()
axes[1, 0].grid()


# ------------------------------------------------------------
# Plot 4: Example Time Series
# ------------------------------------------------------------

axes[1, 1].plot(
    X_test[0].flatten(),
    label="Input Sequence"
)

axes[1, 1].axhline(
    y_test[0],
    label="Actual Target"
)

axes[1, 1].axhline(
    y_pred[0][0],
    label="Predicted Target"
)

axes[1, 1].set_xlabel("Timestep")
axes[1, 1].set_ylabel("Value")
axes[1, 1].set_title("Example Time Series Prediction")

axes[1, 1].legend()
axes[1, 1].grid()


plt.tight_layout()
plt.show()


# ============================================================
# END OF EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 7 COMPLETED")
print("=" * 60)
