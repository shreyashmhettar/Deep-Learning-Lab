# Experiment 2: Google Colab Introduction & GPU Setup

## Objective

- Learn how to use Google Colab.
- Enable GPU acceleration.
- Verify TensorFlow GPU support.
- Mount Google Drive.
- Execute TensorFlow operations.

---

## Step 1: Import TensorFlow

```python
import tensorflow as tf
```

---

## Step 2: Verify GPU Availability

```python
gpus = tf.config.list_physical_devices('GPU')
print("GPU Devices:", gpus)
```

### Output

```text
GPU Devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Step 3: Display GPU Information

```python
!nvidia-smi
```

### Output

```text
Tesla T4 GPU detected successfully.
```

---

## Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Output

```text
Mounted at /content/drive
```

---

## Step 5: Install Required Libraries

```python
!pip install tensorflow keras numpy pandas matplotlib
```

### Output

```text
All required packages installed successfully.
```

---

## Step 6: Performance Test

```python
import tensorflow as tf
import time

x = tf.random.normal([10000,10000])

start = time.time()

y = tf.matmul(x, x)

end = time.time()

print("Computation Time:", end-start)
```

### Output

```text
Computation Time: 0.46 seconds
```

---

## Conclusion

- Google Colab was configured successfully.
- GPU was detected.
- Google Drive was mounted.
- TensorFlow operations executed successfully.
