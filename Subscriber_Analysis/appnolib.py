import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes, stores the predicted class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        # Stopping criteria
        if (depth == self.max_depth) or (np.all(y == y[0])):
            return Node(value=predicted_class)

        # Find the best split
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                gini = self._gini_impurity(y[left_indices], y[~left_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        left_node = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._grow_tree(X[~left_indices], y[~left_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _gini_impurity(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)
        p_left = len(left_labels) / total_samples
        p_right = len(right_labels) / total_samples

        gini_left = 1.0 - sum((np.sum(left_labels == c) / len(left_labels)) ** 2 for c in range(self.num_classes))
        gini_right = 1.0 - sum((np.sum(right_labels == c) / len(right_labels)) ** 2 for c in range(self.num_classes))

        return p_left * gini_left + p_right * gini_right

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
data = pd.read_csv("customer_churn.csv")
imputer = SimpleImputer(strategy='mean')
X = data.drop(columns=['Churn'])
y = data['Churn']
# Convert categorical variables to dummy variables if needed
X = pd.get_dummies(X)
X = imputer.fit_transform(X)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)
# Create and train the Decision Tree classifier
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = tree.predict(X_test)
# Evaluate performance
accuracy = np.mean(y_pred == y_test)
print("Decision Tree Accuracy:", accuracy)
