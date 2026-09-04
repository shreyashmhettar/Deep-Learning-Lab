# Experiment 9: RNN with GRU

## Aim

To understand the GRU (Gated Recurrent Unit) architecture and implement it for sequence prediction. The experiment also compares GRU with LSTM and Simple RNN in terms of performance and computational efficiency.

## Objectives

- Understand GRU architecture.
- Implement GRU for sequence prediction.
- Compare GRU, LSTM, and Simple RNN.
- Analyze training and validation loss.
- Compare prediction performance using RMSE.
- Understand computational efficiency and model complexity.

## Dataset

A synthetic stock price time series dataset is used.

Dataset details:

- Base price: 100
- Number of samples: 500
- Sequence length: 60 timesteps
- Trend and random noise are added to simulate stock price movements.
- Data is normalized using MinMaxScaler.

## Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow
- Keras
- Scikit-learn

## Models Used

### 1. GRU

Architecture:

Input (60 timesteps, 1 feature)

↓

GRU (100 units)

↓

Dropout (0.2)

↓

GRU (50 units)

↓

Dropout (0.2)

↓

Dense (25 units, ReLU)

↓

Dense (1)

### 2. LSTM

Architecture:

Input (60 timesteps, 1 feature)

↓

LSTM (100 units)

↓

Dropout (0.2)

↓

LSTM (50 units)

↓

Dropout (0.2)

↓

Dense (25 units, ReLU)

↓

Dense (1)

### 3. Simple RNN

Architecture:

Input (60 timesteps, 1 feature)

↓

SimpleRNN (100 units)

↓

Dropout (0.2)

↓

SimpleRNN (50 units)

↓

Dropout (0.2)

↓

Dense (25 units, ReLU)

↓

Dense (1)

## Implementation

The main Python file is:

`experiment9_gru.py`

The program performs the following steps:

1. Generates synthetic stock price data.
2. Normalizes the data using MinMaxScaler.
3. Splits the data into training and testing sets.
4. Builds the GRU model.
5. Trains the GRU model.
6. Builds an LSTM model for comparison.
7. Builds a Simple RNN model for comparison.
8. Generates predictions from all three models.
9. Calculates RMSE for each model.
10. Visualizes training loss, validation loss, predictions, and RMSE comparison.

## Results

The experiment produces four main visualizations:

- Training Loss Comparison
- Validation Loss Comparison
- GRU vs LSTM vs RNN Predictions
- RMSE Comparison

The obtained results show that GRU, LSTM, and Simple RNN have similar performance on the synthetic time-series dataset.

GRU provides competitive prediction performance while using a simpler architecture than LSTM.

## GRU vs LSTM vs RNN

| Aspect | Simple RNN | LSTM | GRU |
|---|---|---|---|
| Vanishing Gradient | Poor | Better | Better |
| Parameters | Fewer | More | Medium |
| Training Speed | Fast | Slow | Medium/Fast |
| Long Sequences | Poor | Good | Good |
| Complexity | Low | High | Medium |
| Gates | None | Multiple Gates | Update and Reset Gates |

## Conclusion

This experiment demonstrates the use of GRU networks for sequence prediction and compares their performance with LSTM and Simple RNN models.

GRU provides a good balance between model complexity and performance. It is simpler than LSTM while still being capable of handling sequential dependencies effectively.

## Expected Outcomes

- GRU performance should be similar to LSTM.
- GRU can train faster than LSTM in many cases.
- GRU and LSTM handle longer sequences better than a basic RNN.
- The experiment demonstrates the performance and efficiency trade-off between different recurrent neural network architectures.

## File Structure

Experiment-9/

├── experiment9_gru.py

└── README.md

## Author

**Shreyash Uday Mhettar**

Deep Learning Laboratory
