"""
Multi-layer perceptron classifier.
"""

from sklearn.neural_network import MLPClassifier

def train(x_train, y_train, params):
    """
    Train a multi-layer perceptron classifier.
    """
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)

    return mlp

def classify(model, test_data):
    """
    Classify test data using a trained multi-layer perceptron classifier.
    """
    y_pred = model.predict(test_data)

    return y_pred
