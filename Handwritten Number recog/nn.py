import random
import numpy as np
import matplotlib.pyplot as plt


# Load datasets
X_train = np.loadtxt("X_train.csv", delimiter=",").T
Y_train = np.loadtxt("Y_train.csv", delimiter=",").T
X_test = np.loadtxt("X_test.csv", delimiter=",").T
Y_test = np.loadtxt("Y_test.csv", delimiter=",").T

index = random.randrange(0, X_train.shape[1])


# Activation functions
def hyperbolic_tangent(x):
    return np.tanh(x)


def rectified_linear_unit(x):
    return np.maximum(0, x)


def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx, axis=0)


# Derivatives
def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


# Parameter initialization
def initialize_parameters(nx, nh, ny):
    W1 = np.random.randn(nh, nx) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(ny, nh) * 0.01
    b2 = np.zeros((ny, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# Forward propagation
def forward_propagation(x, parameters):
    W1, b1, W2, b2 = parameters.values()

    Z1 = np.dot(W1, x) + b1
    A1 = hyperbolic_tangent(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    forward_cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return forward_cache


# Cost function
def compute_cost(A2, y):
    m = y.shape[1]
    cost = -(1 / m) * np.sum(y * np.log(A2))
    return cost


# Backward propagation
def backward_propagation(x, y, parameters, forward_cache):
    W1, b1, W2, b2 = parameters.values()

    A1, A2 = forward_cache["A1"], forward_cache["A2"]
    m = x.shape[1]

    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = (1 / m) * np.dot(W2.T, dZ2) * derivative_tanh(A1)
    dW1 = (1 / m) * np.dot(dZ1, x.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients


# Update parameters
def update_parameters(parameters, gradients, learning_rate):
    W1, b1, W2, b2 = parameters.values()
    dW1, db1, dW2, db2 = gradients.values()

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# Neural network model
def neural_network_model(x, y, nh, learning_rate, iterations):
    nx, ny = x.shape[0], y.shape[0]
    cost_list = []

    parameters = initialize_parameters(nx, nh, ny)

    for i in range(iterations):
        forward_cache = forward_propagation(x, parameters)
        cost = compute_cost(forward_cache["A2"], y)
        gradients = backward_propagation(x, y, parameters, forward_cache)
        parameters = update_parameters(parameters, gradients, learning_rate)
        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("Cost after", i, "iterations is:", cost)

    return parameters, cost_list


# Model parameters
iterations = 150
nh = 1000
learning_rate = 0.02
Parameters, Cost_list = neural_network_model(X_train, Y_train, nh=nh, learning_rate=learning_rate, iterations=iterations)


# Accuracy calculation
def calculate_accuracy(inp, labels, parameters):
    forward_cache = forward_propagation(inp, parameters)
    A_out = forward_cache["A2"]

    A_out = np.argmax(A_out, 0)
    labels = np.argmax(labels, 0)

    accuracy = np.mean(A_out == labels) * 100

    return accuracy


print("Accuracy of Train Dataset", calculate_accuracy(X_train, Y_train, Parameters), "%")
print("Accuracy of Test Dataset", calculate_accuracy(X_test, Y_test, Parameters), "%")

# Checking the input
idx = int(random.randrange(0, X_test.shape[1]))
plt.imshow(X_test[:, idx].reshape((28, 28)), cmap='gray')
plt.show()

# Plot graph
t = np.arange(0, iterations)
plt.plot(t, Cost_list)
plt.show()

cache = forward_propagation(X_test[:, idx].reshape(X_test[:, idx].shape[0], 1), Parameters)
A_pred = cache["A2"]
A_pred = np.argmax(A_pred, 0)

print("Model predictor says the written number is: ", A_pred[0])
