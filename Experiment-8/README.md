# EXPERIMENT 8: RNN with LSTM

## Objectives

- Understand LSTM architecture and gates
- Implement LSTM for sequence prediction
- Compare LSTM with Simple RNN
- Handle longer sequences

## Dataset

Synthetic stock price time series with trend and noise.

The dataset contains sequences of 60 timesteps. The models use the
previous sequence of values to predict the next value.

## LSTM Architecture

```text
Input (60 timesteps, 1 feature)
        ↓
LSTM (100 units, return_sequences=True)
        ↓
Dropout (0.2)
        ↓
LSTM (50 units)
        ↓
Dropout (0.2)
        ↓
Dense (25, ReLU)
        ↓
Dense (1) - Output
