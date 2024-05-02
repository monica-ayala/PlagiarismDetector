import xgboost as xgb
import numpy as np

# Create dummy data
X_train = np.random.rand(100, 5)
y_train = np.random.randint(2, size=100)
X_test = np.random.rand(50, 5)
y_test = np.random.randint(2, size=50)

# Fit model
model = xgb.XGBClassifier()
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Starting model training...")
try:
    model.fit(X_train, y_train)
    logging.info("Model training completed successfully.")
except Exception as e:
    logging.error("Error during model training: %s", e)


# Predict and print accuracy
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
