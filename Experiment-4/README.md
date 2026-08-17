# Experiment 4: Multilayer Neural Network

## Objective

- Build a network with multiple hidden layers.
- Understand the impact of network architecture.
- Train the network on multi-class classification.
- Use different activation functions.

## Dataset

**Dataset:** Iris Dataset

- Number of classes: 3
- Number of features: 4
- Classification type: Multi-class classification
- Total samples: 150

## Network Architecture

```text
Input Layer (4 features)
        ↓
Dense Layer (64 neurons, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense Layer (32 neurons, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense Layer (16 neurons, ReLU)
        ↓
Output Layer (3 neurons, Softmax)

## Complete Program

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
...
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# 1. Load and prepare data
iris = load_iris()

X = iris.data
y = to_categorical(iris.target, 3)


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Normalize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Build Multilayer Neural Network
model = Sequential([
    Input(shape=(4,)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])


# 5. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 6. Print architecture
model.summary()


# 7. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)


# 8. Evaluate model
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
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# 9. Predictions
predictions = model.predict(X_test, verbose=0)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)


# 10. Confusion Matrix
cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\nConfusion Matrix:")
print(cm)


# 11. Classification Report
print("\nClassification Report:")

print(
    classification_report(
        true_classes,
        predicted_classes
    )
)


# 12. Visualization
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# Plot Loss
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
axes[0].set_title('Model Loss - Multilayer Network')
axes[0].legend()
axes[0].grid()


# Plot Accuracy
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
axes[1].set_title('Model Accuracy - Multilayer Network')
axes[1].legend()
axes[1].grid()


plt.tight_layout()
plt.show()


Then below that:

```markdown
## Actual Output

Train Accuracy: 0.9833
Test Accuracy: 1.0000

## Confusion Matrix

[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

## Visualization

![Training History](training_history.png)



