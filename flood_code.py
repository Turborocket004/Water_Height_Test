import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path, num_rows, num_cols):
    matrix = np.zeros((num_rows, num_cols))
    row_index = 0
    with open(file_path, 'r') as file:
        for line in file:
            if row_index < num_rows:
                elements = line.strip().split()
                for col_index in range(num_cols):
                    matrix[row_index, col_index] = float(elements[col_index])
                row_index += 1
    return matrix

# Load and normalize training data
training_data_file = 'flood_data_set.txt'
training_data = load_data(training_data_file, 252, 9)
normalized_training_data = training_data / 650
features = normalized_training_data[:, :8]
targets = normalized_training_data[:, 8:]

# Load and normalize test data
test_data_file = 'flood_data_test.txt'
test_data = load_data(test_data_file, 63, 9)
normalized_test_data = test_data / 650
test_features = normalized_test_data[:, :8]
test_targets = normalized_test_data[:, 8:]

# Neural network architecture
hidden_layer_size = 5
input_layer_size = features.shape[1]
output_layer_size = targets.shape[1]

# Initialize weights and biases
np.random.seed(42)
hidden_weights = np.random.rand(input_layer_size, hidden_layer_size)
hidden_biases = np.random.rand(hidden_layer_size)
output_weights = np.random.rand(hidden_layer_size, output_layer_size)
output_biases = np.random.rand(output_layer_size)

# Initialize momentum
prev_output_weights_update = np.zeros_like(output_weights)
prev_output_biases_update = np.zeros_like(output_biases)
prev_hidden_weights_update = np.zeros_like(hidden_weights)
prev_hidden_biases_update = np.zeros_like(hidden_biases)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# Training parameters
num_epochs = 80000
learning_rate = 0.0005
momentum_coefficient = 0.7
epoch_tracker = []
loss_tracker = []

print("Initial hidden biases:\n", hidden_biases)
print("Initial hidden weights:\n", hidden_weights)
print("Initial output biases:\n", output_biases)
print("Initial output weights:\n", output_weights)
print("************************************")

for epoch in range(num_epochs):
    # Forward pass
    hidden_input = np.dot(features, hidden_weights) + hidden_biases
    hidden_output = sigmoid_activation(hidden_input)
    output_input = np.dot(hidden_output, output_weights) + output_biases
    predictions = output_input

    # Compute loss (mean squared error)
    mse_loss = np.mean((predictions - targets) ** 2)
    mse_loss = round(mse_loss, 8)

    # Backward pass
    output_error = predictions - targets
    output_gradient = output_error
    output_weights_update = (learning_rate * np.dot(hidden_output.T, output_gradient)) + (momentum_coefficient * prev_output_weights_update)
    output_biases_update = (learning_rate * np.sum(output_gradient, axis=0)) + (momentum_coefficient * prev_output_biases_update)
    output_weights -= output_weights_update
    output_biases -= output_biases_update

    hidden_error = np.dot(output_gradient, output_weights.T) * sigmoid_derivative(hidden_input)
    hidden_weights_update = (learning_rate * np.dot(features.T, hidden_error)) + (momentum_coefficient * prev_hidden_weights_update)
    hidden_biases_update = (learning_rate * np.sum(hidden_error, axis=0)) + (momentum_coefficient * prev_hidden_biases_update)
    hidden_weights -= hidden_weights_update
    hidden_biases -= hidden_biases_update

    # Update momentum
    prev_output_weights_update = output_weights_update
    prev_output_biases_update = output_biases_update
    prev_hidden_weights_update = hidden_weights_update
    prev_hidden_biases_update = hidden_biases_update

    if epoch % 100 == 0:
        epoch_tracker.append(epoch)
        loss_tracker.append(mse_loss)

print("Final loss:", mse_loss)
scaled_predictions = predictions * 650
print("Predicted output (scaled):\n", scaled_predictions)

# Final weights and biases
print("************************************\n")
print("Final hidden biases:\n", hidden_biases)
print("Final hidden weights:\n", hidden_weights)
print("Final output biases:\n", output_biases)
print("Final output weights:\n", output_weights)

# Forward pass on test data
test_hidden_input = np.dot(test_features, hidden_weights) + hidden_biases
test_hidden_output = sigmoid_activation(test_hidden_input)
test_output_input = np.dot(test_hidden_output, output_weights) + output_biases
test_predictions = test_output_input

# Compute test loss
test_loss = np.mean((test_predictions - test_targets) ** 2)
test_loss = round(test_loss, 8)

# Plotting
plt.figure(figsize=(8, 6))

# Loss vs Epochs
plt.subplot(2, 1, 1)
plt.plot(epoch_tracker, loss_tracker)
plt.xlabel("Epoch")
plt.ylabel("LOSS")
plt.ylim(0.0001, 0.009)
plt.title(f"At Lrate {learning_rate} & Hidden {hidden_layer_size} & Momentum = {momentum_coefficient}")

# True vs Predicted Output with Loss
plt.subplot(2, 1, 2)
plt.plot(test_targets, label="Desired Output", marker='x', color='black')
plt.plot(test_predictions, label="Processed Output", marker='x', color='blue')
plt.title(f"Desired Output vs Predicted Output - Test Loss: {test_loss}")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()