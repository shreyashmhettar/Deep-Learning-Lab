# Experiment 7: Simple RNN

## Objective

- Understand RNN architecture and sequence processing
- Build a Simple RNN for sequence prediction
- Train the model on sequential time-series data
- Visualize model performance and predictions

## Dataset

A synthetic sine wave time-series dataset is used.

- Number of samples: 1000
- Sequence length: 50 timesteps
- Features: 1
- Training samples: 800
- Testing samples: 200
- Task: Time-series regression

## RNN Architecture

```text
Input Sequence
(50 timesteps, 1 feature)
        ↓
SimpleRNN
64 units + ReLU
return_sequences=True
        ↓
SimpleRNN
32 units + ReLU
        ↓
Dropout
0.2
        ↓
Dense
16 units + ReLU
        ↓
Dense
1 unit
        ↓
Predicted Value
