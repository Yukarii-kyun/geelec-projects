import numpy as np

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def predict(self, x):
        z = np.dot(self.weights[1:], x) + self.weights[0]  # Weighted sum + bias
        return self.activation(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights[1:] += update * xi
                self.weights[0] += update  # Update bias

# Example usage
if __name__ == "__main__":
    # Sample data: points (x1, x2) and labels (0 or 1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate logic

    perceptron = SimplePerceptron(input_size=2)
    perceptron.fit(X, y)

    # Test predictions
    for xi in X:
        print(f"Input: {xi}, Prediction: {perceptron.predict(xi)}")