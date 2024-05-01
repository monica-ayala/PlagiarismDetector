from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

X_train = np.load("dataset/X_train.npy")
X_test = np.load("dataset/X_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

def random_forest_model():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = rf_model.score(X_test, y_test)
    print(f"Accuracy (Random Forest): {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
def xgboost_model():
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = xgb_model.score(X_test, y_test)
    print(f"Accuracy (XGBoost): {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
 
xgboost_model()
random_forest_model()