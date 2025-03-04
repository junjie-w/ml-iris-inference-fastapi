import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Data shape: {np.shape(X)}")
    print("First 3 samples:")
    print(np.round(X[:3], 2))
    
    feature_means = np.mean(X, axis=0)
    print(f"Feature means: {feature_means}")
    
    feature_min = np.min(X, axis=0)
    feature_max = np.max(X, axis=0)
    print(f"Feature ranges: {np.round(feature_max - feature_min, 2)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    with open("iris_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved to iris_model.pkl")
    
    return model

if __name__ == "__main__":
    train_model()