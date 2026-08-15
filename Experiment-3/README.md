# Experiment 3: Simple Perceptron (Single Layer Neural Network)

## Objective

* Understand perceptron architecture.
* Implement a single neuron with an activation function.
* Train the model on a binary classification dataset.
* Visualize the decision boundary.

## Dataset

**Dataset:** Iris Dataset

* Classes used: 2
* Samples used: First 100 samples
* Features used: First 2 features
* Task: Binary Classification

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

## Implementation

### 1. Load Dataset

```python
iris = load_iris()

X = iris.data[:100, :2]
y = iris.target[:100]
```

### 2. Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 3. Normalize Features

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Create Perceptron

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(1, activation='sigmoid')
])
```

### 5. Compile Model

```python
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 6. Train Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
```

### 7. Evaluate Model

```python
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

print("===== RESULTS =====")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```

## Complete Program

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load Iris dataset
iris = load_iris()

X = iris.data[:100, :2]
y = iris.target[:100]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Normalize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create Perceptron
model = Sequential([
    Input(shape=(2,)),
    Dense(1, activation='sigmoid')
])


# Compile model
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)


# Evaluate
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


# Visualize decision boundary
h = 0.02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = model.predict(
    np.c_[xx.ravel(), yy.ravel()],
    verbose=0
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(
    xx,
    yy,
    Z,
    cmap=plt.cm.RdBu,
    alpha=0.6
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.RdBu,
    edgecolors='k'
)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')

plt.show()


# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')


plt.subplot(1, 2, 2)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.tight_layout()
plt.show()
```

## Actual Output

```text
===== RESULTS =====
Train Accuracy: 0.9750
Test Accuracy: 1.0000
```

### Accuracy

* Training Accuracy: **97.50%**
* Testing Accuracy: **100%**

## Visualization Output

### Perceptron Decision Boundary

The decision-boundary visualization shows the two Iris classes and the linear boundary learned by the perceptron.

### Model Loss

The training and validation loss decrease over the epochs, showing that the model learns during training.

### Model Accuracy

The training accuracy increases to approximately 97.5%, while the validation accuracy reaches 100%.

## Key Concepts

| Concept             | Used In Experiment                |
| ------------------- | --------------------------------- |
| Activation Function | Sigmoid                           |
| Loss Function       | Binary Crossentropy               |
| Optimizer           | SGD (Stochastic Gradient Descent) |
| Neural Network      | Single-layer Perceptron           |
| Classification      | Binary Classification             |

## Result

The single-layer perceptron was successfully implemented and trained on the Iris dataset. The model achieved **97.50% training accuracy** and **100% test accuracy**.

## Conclusion

The experiment successfully demonstrated the working of a simple perceptron for binary classification. The decision boundary and training graphs were visualized, and the model achieved high classification accuracy.
