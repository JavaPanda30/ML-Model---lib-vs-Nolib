import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, len(np.unique(y))))
        classes = np.unique(y)
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - self._label_encode(y, classes))) / m
            self.theta -= self.learning_rate * gradient
            
    def predict(self, X):
        z = np.dot(X, self.theta)
        probabilities = self.sigmoid(z)
        predictions = np.argmax(probabilities, axis=1)  
        return predictions

    def _label_encode(self, y, classes):
        label_encoded = np.zeros((len(y), len(classes)))
        for i, cls in enumerate(classes):
            label_encoded[:, i] = (y == cls)
        return label_encoded
    

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_classification_report(y_true, y_pred, labels):
    report = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    for label in labels:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
    
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score =( 2 * (precision * recall) )/ (precision + recall)
        support = np.sum(y_true == label)
        
        report['precision'].append(precision)
        report['recall'].append(recall)
        report['f1-score'].append(f1_score)
        report['support'].append(support)
    
    return pd.DataFrame(report, index=labels)


data = pd.read_csv("Weather_Prediction\weather-datafile.csv")  
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['weather']

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)
classes = np.unique(y)
predicted_labels = [classes[pred] for pred in predictions]
# Calculate accuracy
accuracy = calculate_accuracy(y, predicted_labels)
print("Accuracy:", accuracy)
report = calculate_classification_report(y,predicted_labels,classes)
print("Classification Report:")
print(report)
