import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical


# ============================================================
# 1. Load and Prepare MNIST Dataset
# ============================================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# ============================================================
# 2. Build CNN
# ============================================================

model = Sequential([
    
    # First Convolutional Block
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),

    # Fully Connected Layers
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# ============================================================
# 3. Compile Model
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 4. Display Model Summary
# ============================================================

model.summary()


# ============================================================
# 5. Train Model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# ============================================================
# 6. Evaluate Model
# ============================================================

train_loss, train_acc = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===== RESULTS =====")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# ============================================================
# 7. Visualize Training History
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# Loss
axes[0].plot(
    history.history['loss'],
    label='Training Loss'
)

axes[0].plot(
    history.history['val_loss'],
    label='Validation Loss'
)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid()


# Accuracy
axes[1].plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

axes[1].plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid()


plt.tight_layout()
plt.show()


# ============================================================
# 8. Visualize Sample Predictions
# ============================================================

predictions = model.predict(
    X_test[:10],
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

true_labels = np.argmax(
    y_test[:10],
    axis=1
)

plt.figure(figsize=(15, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i].reshape(28, 28),
        cmap='gray'
    )

    plt.title(
        f"True: {true_labels[i]}\nPred: {predicted_labels[i]}"
    )

    plt.axis('off')

plt.suptitle('Sample Predictions')
plt.tight_layout()
plt.show()


# ============================================================
# 9. Visualize Learned Filters
# ============================================================

first_conv_layer = model.layers[0]

filters, biases = first_conv_layer.get_weights()

print("\nFirst Convolution Layer Filter Shape:", filters.shape)

plt.figure(figsize=(12, 6))

for i in range(min(16, filters.shape[3])):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        filters[:, :, 0, i],
        cmap='gray'
    )

    plt.title(f"Filter {i + 1}")
    plt.axis('off')

plt.suptitle('Learned CNN Filters')
plt.tight_layout()
plt.show()


# ============================================================
# 10. Visualize Feature Maps
# ============================================================

feature_model = Model(
    inputs=model.input,
    outputs=model.layers[1].output
)

feature_maps = feature_model.predict(
    X_test[:1],
    verbose=0
)

plt.figure(figsize=(12, 8))

for i in range(min(16, feature_maps.shape[-1])):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        feature_maps[0, :, :, i],
        cmap='gray'
    )

    plt.title(f"Feature Map {i + 1}")
    plt.axis('off')

plt.suptitle('CNN Feature Maps')
plt.tight_layout()
plt.show()