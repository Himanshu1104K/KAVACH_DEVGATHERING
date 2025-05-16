import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing TensorFlow with a fallback to scikit-learn for model creation
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    USE_TENSORFLOW = True
    print("Using TensorFlow for model training")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    USE_TENSORFLOW = False
    print("TensorFlow not available, using Gradient Boosting Regressor instead")

# Set random seeds for reproducibility
np.random.seed(42)
if USE_TENSORFLOW:
    tf.random.set_seed(42)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading and preprocessing the dataset...")

# Load the dataset (using the training subset which has selected features)
df = pd.read_csv("../../Dataset/Efficiency/military_performance_training_data.csv")

# Check available features
print("Dataset shape:", df.shape)
print("Available features:", df.columns.tolist())

# Separate features and target
X = df.drop("Efficiency", axis=1)
y = df["Efficiency"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Transform the data
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Get the shape of processed data
input_dim = X_train_processed.shape[1]
print(f"Input dimension after preprocessing: {input_dim}")

if USE_TENSORFLOW:
    # Define model architecture with TensorFlow
    def create_model(input_dim):
        model = Sequential(
            [
                # Input layer
                Dense(128, activation="relu", input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                # Hidden layers
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                # Output layer for efficiency prediction (regression)
                Dense(1),
            ]
        )

        # Compile the model
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

        return model

    # Create and display the model
    model = create_model(input_dim)
    model.summary()

    # Define callbacks for training
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath="models/efficiency_predictor_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001, verbose=1
        ),
    ]

    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"\nTest Mean Absolute Error: {test_mae:.4f}")

    # Save the final model
    model.save("models/efficiency_predictor_final.keras")
    print("Model saved to 'models/efficiency_predictor_final.keras'")

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    # Plot training & validation mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("Model Mean Absolute Error")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    plt.savefig("results/training_history.png")
    plt.close()

    # Make predictions on test set
    y_pred = model.predict(X_test_processed)

else:
    # Use Gradient Boosting Regressor instead
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbose=1
    )

    print("\nTraining Gradient Boosting model...")
    model.fit(X_train_processed, y_train)

    # Evaluate on validation set
    val_pred = model.predict(X_val_processed)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE: {val_mae:.4f}")

    # Make predictions on test set
    y_pred = model.predict(X_test_processed)
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"\nTest Mean Absolute Error: {test_mae:.4f}")

    # Save the model
    import joblib

    joblib.dump(model, "models/efficiency_predictor_final.joblib")
    print("Model saved to 'models/efficiency_predictor_final.joblib'")

# Save the preprocessor for later use
import joblib

joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Preprocessor saved to 'models/preprocessor.pkl'")

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.title("Actual vs Predicted Efficiency")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png")
plt.close()

# Feature importance analysis
from sklearn.ensemble import RandomForestRegressor

# Combine preprocessor and a simpler model for explanation
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Get feature names after one-hot encoding
feature_names = numerical_cols.copy()
for col in categorical_cols:
    values = X[col].unique()
    for val in values:
        feature_names.append(f"{col}_{val}")

# Get feature importance scores
importances = rf_model.feature_importances_

# Adjust feature names list length to match importances length
feature_names = feature_names[: len(importances)]

# Visualize feature importance (top 15)
feat_imp = (
    pd.DataFrame({"Feature": feature_names, "Importance": importances})
    .sort_values(by="Importance", ascending=False)
    .head(15)
)

plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Top 15 Feature Importance (Random Forest Approximation)")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()


# Create a function for making predictions on new data
def predict_efficiency(
    new_data, model_path=None, preprocessor_path="models/preprocessor.pkl"
):
    """
    Make efficiency predictions on new data

    Parameters:
    -----------
    new_data : pandas DataFrame
        New data with the same features as the training data
    model_path : str
        Path to the saved model (automatically determines model type)
    preprocessor_path : str
        Path to the saved preprocessor

    Returns:
    --------
    numpy array
        Predicted efficiency values
    """
    # Determine model path if not provided
    if model_path is None:
        if os.path.exists("models/efficiency_predictor_final.keras"):
            model_path = "models/efficiency_predictor_final.keras"
            is_tf_model = True
        elif os.path.exists("models/efficiency_predictor_best.keras"):
            model_path = "models/efficiency_predictor_best.keras"
            is_tf_model = True
        else:
            model_path = "models/efficiency_predictor_final.joblib"
            is_tf_model = False
    else:
        is_tf_model = model_path.endswith((".h5", ".keras"))

    print(f"Loading model from {model_path}")

    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess the new data
    X_new_processed = preprocessor.transform(new_data)

    # Load appropriate model type and make predictions
    if is_tf_model:
        model = load_model(model_path)
        predictions = model.predict(X_new_processed)
        return predictions.flatten()
    else:
        model = joblib.load(model_path)
        return model.predict(X_new_processed)


print("\nModel Training and Evaluation Complete!")
print("Results and visualizations saved in the 'results' folder")

# Main execution
if __name__ == "__main__":
    print("Testing prediction functionality...")

    # Load the dataset
    print("Loading test data...")
    df = pd.read_csv("../Dataset/military_performance_training_data.csv")

    # Use the first 5 rows as sample data
    sample_data = df.drop("Efficiency", axis=1).iloc[:5]
    actual_values = df["Efficiency"].iloc[:5].values

    # Make predictions
    print("Making predictions...")
    try:
        sample_predictions = predict_efficiency(sample_data)

        # Compare predictions with actual values
        print("\nPrediction Results:")
        for i, (pred, actual) in enumerate(zip(sample_predictions, actual_values)):
            print(
                f"Sample {i+1}: Predicted={pred:.4f}, Actual={actual:.4f}, Difference={abs(pred-actual):.4f}"
            )

        print("\nPrediction test successful!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
