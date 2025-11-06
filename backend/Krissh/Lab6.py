import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
from collections import Counter

warnings.filterwarnings('ignore')

from backend.Krissh.preprocessing import preprocessing

merged_df = preprocessing()

class NaiveBayes:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}
    
    def fit(self, X, y):
        # Calculate class probabilities P(Class)
        for class_val in y.unique():
            self.class_prob[class_val] = len(y[y == class_val]) / len(y)
        
        # Calculate feature probabilities P(Feature|Class)
        self.feature_prob = {}
        for col in X.columns:
            self.feature_prob[col] = {}
            for class_val in y.unique():
                class_data = X[y == class_val][col]
                self.feature_prob[col][class_val] = {}
                
                for feature_val in X[col].unique():
                    count = len(class_data[class_data == feature_val])
                    # Laplace smoothing to handle zero probabilities
                    prob = (count + 1) / (len(class_data) + len(X[col].unique()))
                    self.feature_prob[col][class_val][feature_val] = prob
    
    def predict(self, X):
        predictions = []
        
        for _, row in X.iterrows():
            class_scores = {}
            
            # Calculate score for each class
            for class_val in self.class_prob.keys():
                score = self.class_prob[class_val]
                
                # Multiply by feature probabilities
                for col in X.columns:
                    feature_val = row[col]
                    if feature_val in self.feature_prob[col][class_val]:
                        score *= self.feature_prob[col][class_val][feature_val]
                    else:
                        # Handle unseen features with small probability
                        score *= 0.0001
                
                class_scores[class_val] = score
            
            # Pick class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return predictions


# Use the model
def run_naive_bayes(merged_df):
    # Select features and target
    features = ['TeamName', 'DriverCode', 'Position']
    target = 'Status'
    
    # Clean data - remove missing values
    df_clean = merged_df[features + [target]].dropna()
    
    # Split into train and test (70-30 split)
    train_size = int(0.7 * len(df_clean))
    train_data = df_clean[:train_size]
    test_data = df_clean[train_size:]
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Train the model
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    
    # Make predictions
    predictions = nb.predict(X_test)
    
    # Calculate accuracy
    correct = sum([1 for i in range(len(predictions)) if predictions[i] == y_test.values[i]])
    accuracy = correct / len(predictions)

    f_string = ", ".join(i for i in features)
    print(f"Features: {f_string}")
    print(f"Target: {target}")
    print(f"Total Test Samples: {len(predictions)}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return nb


# Run the model
model = run_naive_bayes(merged_df)