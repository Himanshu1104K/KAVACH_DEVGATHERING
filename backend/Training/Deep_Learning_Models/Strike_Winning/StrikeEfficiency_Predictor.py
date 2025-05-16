import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

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

# Function to load and prepare dataset
def prepare_dataset(filepath):
    """
    Load and prepare the dataset for strike efficiency prediction
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file with soldier data
        
    Returns:
    --------
    X : DataFrame
        Features for prediction
    y : Series
        Target variable (strike efficiency/winning rate)
    """
    print(f"Loading dataset from {filepath}...")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Print basic information
    print("Dataset shape:", df.shape)
    print("Available features:", df.columns.tolist())
    
    # Separate features and target (assuming column name is 'StrikeEfficiency' or 'WinningRate')
    target_column = None
    for col in ['StrikeEfficiency', 'WinningRate', 'Efficiency']:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        raise ValueError("Target column not found in dataset. Expected 'StrikeEfficiency', 'WinningRate', or 'Efficiency'")
        
    print(f"Using '{target_column}' as target variable")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y

def train_strike_efficiency_model(X, y):
    """
    Train a model to predict strike efficiency
    
    Parameters:
    -----------
    X : DataFrame
        Features for prediction
    y : Series
        Target variable (strike efficiency/winning rate)
        
    Returns:
    --------
    model : trained model
    preprocessor : fitted preprocessor
    metrics : dict
        Performance metrics
    """
    # Feature engineering - create additional features
    print("Performing feature engineering...")
    
    # Make a copy to avoid modifying the original dataframe
    X_enhanced = X.copy()
    
    # Create new features if they don't already exist
    if 'BMI' in X_enhanced.columns and 'Weight' in X_enhanced.columns and 'Height' in X_enhanced.columns:
        # BMI already exists
        pass
    elif 'Weight' in X_enhanced.columns and 'Height' in X_enhanced.columns:
        # Calculate BMI if it doesn't exist but we have weight and height
        X_enhanced['BMI'] = X_enhanced['Weight'] / ((X_enhanced['Height']/100) ** 2)
    
    # Combat effectiveness score (combine strength, speed, and endurance if they exist)
    if all(col in X_enhanced.columns for col in ['Strength', 'Speed', 'Endurance']):
        X_enhanced['CombatEffectiveness'] = (X_enhanced['Strength'] * 0.4 + 
                                           X_enhanced['Speed'] * 0.3 + 
                                           X_enhanced['Endurance'] * 0.3)
    
    # Physical condition score (inverse of fatigue and stress)
    if all(col in X_enhanced.columns for col in ['Fatigue', 'Stress']):
        max_fatigue = X_enhanced['Fatigue'].max()
        max_stress = X_enhanced['Stress'].max()
        X_enhanced['PhysicalCondition'] = 1 - ((X_enhanced['Fatigue'] / max_fatigue * 0.6) + 
                                             (X_enhanced['Stress'] / max_stress * 0.4))
    
    # Experience-Training ratio
    if all(col in X_enhanced.columns for col in ['CombatExperience', 'TrainingHours']):
        # Normalize TrainingHours to avoid division by large numbers
        X_enhanced['ExperienceTrainingRatio'] = X_enhanced['CombatExperience'] / (X_enhanced['TrainingHours'] / 1000 + 0.001)
    
    # Weapon proficiency to strength ratio for weapon handling
    if all(col in X_enhanced.columns for col in ['WeaponProficiency', 'Strength']):
        X_enhanced['WeaponHandlingEfficiency'] = X_enhanced['WeaponProficiency'] / (X_enhanced['Strength'] + 0.001)
    
    # Log transform for highly skewed features
    skewed_columns = ['TrainingHours'] if 'TrainingHours' in X_enhanced.columns else []
    for col in skewed_columns:
        X_enhanced[f'{col}_log'] = np.log1p(X_enhanced[col])
    
    # Interaction terms between categorical and numerical features
    if 'Terrain' in X_enhanced.columns and 'WeaponType' in X_enhanced.columns:
        # Create terrain-specific one-hot encoding
        X_enhanced['Urban_Combat'] = (X_enhanced['Terrain'] == 'Urban').astype(int)
        X_enhanced['Desert_Combat'] = (X_enhanced['Terrain'] == 'Desert').astype(int)
        X_enhanced['Forest_Combat'] = (X_enhanced['Terrain'] == 'Forest').astype(int)
        X_enhanced['Mountain_Combat'] = (X_enhanced['Terrain'] == 'Mountain').astype(int)
        
        # Create weapon-specific one-hot encoding
        X_enhanced['Rifle_User'] = X_enhanced['WeaponType'].str.contains('Rifle').astype(int)
        X_enhanced['Machine_Gun_User'] = (X_enhanced['WeaponType'] == 'Machine Gun').astype(int)
        X_enhanced['Shotgun_User'] = (X_enhanced['WeaponType'] == 'Shotgun').astype(int)
        X_enhanced['Handgun_User'] = (X_enhanced['WeaponType'] == 'Handgun').astype(int)
    
    # Identify numerical and categorical columns
    numerical_cols = X_enhanced.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_enhanced.select_dtypes(include=["object"]).columns.tolist()

    print(f"Enhanced feature set:")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Total features: {len(numerical_cols) + len(categorical_cols)}")

    # Create preprocessing pipeline with robust scaling for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),  # Use RobustScaler instead of StandardScaler
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_enhanced, y, test_size=0.3, random_state=42, stratify=None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
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
        # Define model architecture with TensorFlow - Enhanced for better accuracy
        model = Sequential([
            # Input layer
            Dense(256, activation="relu", input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.25),
            # Hidden layers
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            Dense(64, activation="relu"), 
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            # Output layer for efficiency prediction (regression)
            Dense(1, activation="sigmoid"),  # Sigmoid for probability output (win rate)
        ])

        # Compile the model with a lower learning rate for more stable training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
            loss="binary_crossentropy" if y.nunique() <= 2 else "mean_squared_error", 
            metrics=["accuracy" if y.nunique() <= 2 else "mae"]
        )
        
        model.summary()

        # Define callbacks for training
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                filepath="models/strike_efficiency_best.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=0.00005, verbose=1
            ),
        ]

        # Train the model with a smaller batch size for better generalization
        print("\nTraining the model...")
        history = model.fit(
            X_train_processed,
            y_train,
            validation_data=(X_val_processed, y_val),
            epochs=150,  # More epochs with early stopping
            batch_size=16,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate on test set
        if y.nunique() <= 2:
            test_loss, test_acc = model.evaluate(X_test_processed, y_test, verbose=0)
            print(f"\nTest Accuracy: {test_acc:.4f}")
            metrics = {"accuracy": test_acc, "loss": test_loss}
        else:
            test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
            print(f"\nTest Mean Absolute Error: {test_mae:.4f}")
            metrics = {"mae": test_mae, "loss": test_loss}

        # Save the final model
        model.save("models/strike_efficiency_final.keras")
        print("Model saved to 'models/strike_efficiency_final.keras'")

        # Train additional models for ensembling
        print("\nTraining ensemble models for better predictions...")
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        # Train a Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        rf_model.fit(X_train_processed, y_train)
        
        # Train a Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_processed, y_train)
        
        # Save these models
        import joblib
        joblib.dump(rf_model, "models/strike_efficiency_rf.joblib")
        joblib.dump(gb_model, "models/strike_efficiency_gb.joblib")
        
        # Make ensemble predictions on validation set
        nn_val_preds = model.predict(X_val_processed).flatten()
        rf_val_preds = rf_model.predict(X_val_processed)
        gb_val_preds = gb_model.predict(X_val_processed)
        
        # Find optimal weights for ensemble using validation performance
        from scipy.optimize import minimize
        
        def ensemble_error(weights):
            # Convert to array for matrix operations
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights to sum to 1
            
            # Weighted average of predictions
            ensemble_preds = (weights[0] * nn_val_preds +
                              weights[1] * rf_val_preds +
                              weights[2] * gb_val_preds)
            
            # Calculate error
            return mean_absolute_error(y_val, ensemble_preds)
        
        # Initial equal weights
        initial_weights = [1/3, 1/3, 1/3]
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Optimize weights
        result = minimize(ensemble_error, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = result['x']
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
        
        print(f"Optimal ensemble weights: NN={optimal_weights[0]:.3f}, RF={optimal_weights[1]:.3f}, GB={optimal_weights[2]:.3f}")
        
        # Save optimal weights
        np.save("models/ensemble_weights.npy", optimal_weights)
        
        # Make ensemble predictions on test set
        nn_test_preds = model.predict(X_test_processed).flatten()
        rf_test_preds = rf_model.predict(X_test_processed)
        gb_test_preds = gb_model.predict(X_test_processed)
        
        ensemble_test_preds = (optimal_weights[0] * nn_test_preds +
                              optimal_weights[1] * rf_test_preds +
                              optimal_weights[2] * gb_test_preds)
        
        # Calculate ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_test_preds)
        ensemble_mse = mean_squared_error(y_test, ensemble_test_preds)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_r2 = r2_score(y_test, ensemble_test_preds)
        
        print("\nEnsemble Model Performance:")
        print(f"Ensemble MAE: {ensemble_mae:.4f} (vs NN MAE: {test_mae:.4f})")
        print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"Ensemble R²: {ensemble_r2:.4f}")
        
        # Update metrics with ensemble results
        metrics.update({
            "ensemble_mae": ensemble_mae,
            "ensemble_rmse": ensemble_rmse,
            "ensemble_r2": ensemble_r2
        })

        # Plot training history
        plt.figure(figsize=(12, 5))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        # Plot training & validation accuracy/MAE
        plt.subplot(1, 2, 2)
        metric_key = "accuracy" if y.nunique() <= 2 else "mae"
        plt.plot(history.history[metric_key])
        plt.plot(history.history[f"val_{metric_key}"])
        plt.title(f"Model {'Accuracy' if y.nunique() <= 2 else 'MAE'}")
        plt.ylabel(metric_key.upper())
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        plt.tight_layout()
        plt.savefig("results/strike_efficiency_training_history.png")
        plt.close()

        # Make predictions on test set
        y_pred = model.predict(X_test_processed)
        if y.nunique() <= 2:
            y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary

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
        metrics = {"mae": test_mae}

        # Save the model
        import joblib
        joblib.dump(model, "models/strike_efficiency_final.joblib")
        print("Model saved to 'models/strike_efficiency_final.joblib'")

    # Save the preprocessor for later use
    import joblib
    joblib.dump(preprocessor, "models/strike_efficiency_preprocessor.pkl")
    print("Preprocessor saved to 'models/strike_efficiency_preprocessor.pkl'")

    # Calculate metrics
    if y.nunique() > 2:  # Regression metrics for continuous target
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nModel Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        metrics.update({"rmse": rmse, "r2": r2})
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual Strike Efficiency")
        plt.ylabel("Predicted Strike Efficiency")
        plt.title("Actual vs Predicted Strike Efficiency")
        plt.tight_layout()
        plt.savefig("results/strike_efficiency_actual_vs_predicted.png")
        plt.close()
    
    # Feature importance analysis
    if categorical_cols or numerical_cols:
        print("\nAnalyzing feature importance...")
        from sklearn.ensemble import RandomForestRegressor

        # Train a Random Forest for feature importance
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
        plt.title("Top 15 Feature Importance for Strike Efficiency")
        plt.tight_layout()
        plt.savefig("results/strike_efficiency_feature_importance.png")
        plt.close()
    
    return model, preprocessor, metrics

def predict_strike_efficiency(
    new_data, model_path=None, preprocessor_path="models/strike_efficiency_preprocessor.pkl",
    use_ensemble=True
):
    """
    Make strike efficiency predictions on new data

    Parameters:
    -----------
    new_data : pandas DataFrame
        New data with the same features as the training data
    model_path : str
        Path to the saved model (automatically determines model type)
    preprocessor_path : str
        Path to the saved preprocessor
    use_ensemble : bool
        Whether to use ensemble prediction (better accuracy)

    Returns:
    --------
    numpy array
        Predicted strike efficiency values
    """
    # Import joblib here to ensure it's available
    import joblib
    import os
    
    # Feature engineering - replicate the same transformations as in training
    X_enhanced = new_data.copy()
    
    # Create the same engineered features as during training
    if 'BMI' in X_enhanced.columns and 'Weight' in X_enhanced.columns and 'Height' in X_enhanced.columns:
        # BMI already exists
        pass
    elif 'Weight' in X_enhanced.columns and 'Height' in X_enhanced.columns:
        # Calculate BMI if it doesn't exist but we have weight and height
        X_enhanced['BMI'] = X_enhanced['Weight'] / ((X_enhanced['Height']/100) ** 2)
    
    # Combat effectiveness score
    if all(col in X_enhanced.columns for col in ['Strength', 'Speed', 'Endurance']):
        X_enhanced['CombatEffectiveness'] = (X_enhanced['Strength'] * 0.4 + 
                                           X_enhanced['Speed'] * 0.3 + 
                                           X_enhanced['Endurance'] * 0.3)
    
    # Physical condition score
    if all(col in X_enhanced.columns for col in ['Fatigue', 'Stress']):
        # Use same logic as training, but be careful with max values
        # For prediction, use reasonable max values similar to training
        max_fatigue = 100  # Reasonable max value
        max_stress = 100  # Reasonable max value
        X_enhanced['PhysicalCondition'] = 1 - ((X_enhanced['Fatigue'] / max_fatigue * 0.6) + 
                                             (X_enhanced['Stress'] / max_stress * 0.4))
    
    # Experience-Training ratio
    if all(col in X_enhanced.columns for col in ['CombatExperience', 'TrainingHours']):
        X_enhanced['ExperienceTrainingRatio'] = X_enhanced['CombatExperience'] / (X_enhanced['TrainingHours'] / 1000 + 0.001)
    
    # Weapon proficiency to strength ratio
    if all(col in X_enhanced.columns for col in ['WeaponProficiency', 'Strength']):
        X_enhanced['WeaponHandlingEfficiency'] = X_enhanced['WeaponProficiency'] / (X_enhanced['Strength'] + 0.001)
    
    # Log transform
    skewed_columns = ['TrainingHours'] if 'TrainingHours' in X_enhanced.columns else []
    for col in skewed_columns:
        X_enhanced[f'{col}_log'] = np.log1p(X_enhanced[col])
    
    # Categorical interactions
    if 'Terrain' in X_enhanced.columns and 'WeaponType' in X_enhanced.columns:
        X_enhanced['Urban_Combat'] = (X_enhanced['Terrain'] == 'Urban').astype(int)
        X_enhanced['Desert_Combat'] = (X_enhanced['Terrain'] == 'Desert').astype(int)
        X_enhanced['Forest_Combat'] = (X_enhanced['Terrain'] == 'Forest').astype(int)
        X_enhanced['Mountain_Combat'] = (X_enhanced['Terrain'] == 'Mountain').astype(int)
        
        X_enhanced['Rifle_User'] = X_enhanced['WeaponType'].str.contains('Rifle').astype(int)
        X_enhanced['Machine_Gun_User'] = (X_enhanced['WeaponType'] == 'Machine Gun').astype(int)
        X_enhanced['Shotgun_User'] = (X_enhanced['WeaponType'] == 'Shotgun').astype(int)
        X_enhanced['Handgun_User'] = (X_enhanced['WeaponType'] == 'Handgun').astype(int)
    
    # Determine model path if not provided
    if model_path is None:
        if os.path.exists("models/strike_efficiency_final.keras"):
            model_path = "models/strike_efficiency_final.keras"
            is_tf_model = True
        elif os.path.exists("models/strike_efficiency_best.keras"):
            model_path = "models/strike_efficiency_best.keras"
            is_tf_model = True
        else:
            model_path = "models/strike_efficiency_final.joblib"
            is_tf_model = False
    else:
        is_tf_model = model_path.endswith((".h5", ".keras"))

    print(f"Loading model from {model_path}")
    
    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess the new data
    X_new_processed = preprocessor.transform(X_enhanced)

    # If using ensemble and ensemble models exist
    if use_ensemble and os.path.exists("models/ensemble_weights.npy") and \
       os.path.exists("models/strike_efficiency_rf.joblib") and \
       os.path.exists("models/strike_efficiency_gb.joblib") and \
       os.path.exists(model_path):
        
        print("Using ensemble prediction for better accuracy")
        
        # Load models and weights
        weights = np.load("models/ensemble_weights.npy")
        rf_model = joblib.load("models/strike_efficiency_rf.joblib")
        gb_model = joblib.load("models/strike_efficiency_gb.joblib")
        
        # Load neural network model
        if is_tf_model:
            from tensorflow.keras.models import load_model
            nn_model = load_model(model_path)
            nn_preds = nn_model.predict(X_new_processed).flatten()
        else:
            nn_model = joblib.load(model_path)
            nn_preds = nn_model.predict(X_new_processed)
        
        # Get predictions from all models
        rf_preds = rf_model.predict(X_new_processed)
        gb_preds = gb_model.predict(X_new_processed)
        
        # Weighted ensemble prediction
        ensemble_preds = (weights[0] * nn_preds +
                          weights[1] * rf_preds +
                          weights[2] * gb_preds)
        
        return ensemble_preds
    else:
        # Regular single model prediction
        if is_tf_model:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            predictions = model.predict(X_new_processed)
            return predictions.flatten()
        else:
            model = joblib.load(model_path)
            return model.predict(X_new_processed)

def create_sample_dataset(output_path, num_samples=1000):
    """
    Create a sample dataset with features that might be relevant for strike efficiency
    
    Parameters:
    -----------
    output_path : str
        Path to save the CSV file
    num_samples : int
        Number of samples to generate
    """
    print(f"Creating sample dataset with {num_samples} samples...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    data = {
        # Physical attributes
        'Age': np.random.randint(18, 45, num_samples),
        'Weight': np.random.normal(75, 10, num_samples),  # in kg
        'Height': np.random.normal(175, 10, num_samples),  # in cm
        'BMI': np.random.normal(24, 3, num_samples),
        
        # Physical fitness
        'Strength': np.random.normal(70, 15, num_samples),
        'Speed': np.random.normal(65, 15, num_samples),
        'Endurance': np.random.normal(60, 15, num_samples),
        'Agility': np.random.normal(55, 15, num_samples),
        
        # Combat training
        'TrainingHours': np.random.randint(50, 5000, num_samples),
        'CombatExperience': np.random.randint(0, 10, num_samples),
        'ReactionTime': np.random.normal(0.25, 0.05, num_samples),  # in seconds
        
        # Tactical knowledge
        'TacticalKnowledge': np.random.normal(60, 15, num_samples),
        
        # Physiological state
        'Fatigue': np.random.normal(30, 10, num_samples),
        'Stress': np.random.normal(40, 15, num_samples),
        'SleepHours': np.random.normal(6.5, 1.0, num_samples),
        
        # Weapon proficiency
        'WeaponAccuracy': np.random.normal(70, 15, num_samples),
        'WeaponHandlingSpeed': np.random.normal(65, 15, num_samples),
        
        # Environmental factors
        'Temperature': np.random.normal(25, 5, num_samples),
        'Terrain': np.random.choice(['Urban', 'Forest', 'Desert', 'Mountain'], num_samples),
        'TimeOfDay': np.random.choice(['Day', 'Night'], num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate Strike Efficiency (a complex function of the above features)
    # This is a simplified model - in reality, this would be based on actual performance data
    
    # Normalize some key contributors between 0 and 1
    strength_norm = (df['Strength'] - df['Strength'].min()) / (df['Strength'].max() - df['Strength'].min())
    speed_norm = (df['Speed'] - df['Speed'].min()) / (df['Speed'].max() - df['Speed'].min())
    training_norm = (df['TrainingHours'] - df['TrainingHours'].min()) / (df['TrainingHours'].max() - df['TrainingHours'].min())
    fatigue_rev_norm = 1 - (df['Fatigue'] - df['Fatigue'].min()) / (df['Fatigue'].max() - df['Fatigue'].min())  # Reverse scale
    stress_rev_norm = 1 - (df['Stress'] - df['Stress'].min()) / (df['Stress'].max() - df['Stress'].min())  # Reverse scale
    accuracy_norm = (df['WeaponAccuracy'] - df['WeaponAccuracy'].min()) / (df['WeaponAccuracy'].max() - df['WeaponAccuracy'].min())
    
    # Terrain modifier (example)
    terrain_mod = pd.Series(1.0, index=df.index)  # Default modifier
    terrain_mod[df['Terrain'] == 'Urban'] = 1.1  # Better in urban
    terrain_mod[df['Terrain'] == 'Desert'] = 0.9  # Worse in desert
    
    # Time of day modifier (example)
    time_mod = pd.Series(1.0, index=df.index)  # Default modifier
    time_mod[df['TimeOfDay'] == 'Night'] = 0.85  # Worse at night
    
    # Calculate efficiency as weighted sum of normalized factors with terrain and time modifiers
    weights = [0.2, 0.15, 0.25, 0.1, 0.1, 0.2]  # Example weights
    
    base_efficiency = (
        weights[0] * strength_norm +
        weights[1] * speed_norm +
        weights[2] * training_norm +
        weights[3] * fatigue_rev_norm +
        weights[4] * stress_rev_norm +
        weights[5] * accuracy_norm
    )
    
    # Apply modifiers and add some random noise
    efficiency = base_efficiency * terrain_mod * time_mod + np.random.normal(0, 0.05, num_samples)
    
    # Clip to ensure values are between 0 and 1
    efficiency = np.clip(efficiency, 0, 1)
    
    # Add to dataframe
    df['StrikeEfficiency'] = efficiency
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")
    
    return df

# Main execution
if __name__ == "__main__":
    print("Strike Efficiency Predictor")
    print("--------------------------")
    
    # Check if dataset exists, if not create a sample one
    dataset_path = "../../Dataset/Strike_Winning/soldier_strike_efficiency.csv"
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Creating a sample dataset.")
        os.makedirs("../Dataset", exist_ok=True)
        create_sample_dataset(dataset_path)
    
    # Load and prepare the dataset
    X, y = prepare_dataset(dataset_path)
    
    # Train the model
    model, preprocessor, metrics = train_strike_efficiency_model(X, y)
    
    print("\nModel Training and Evaluation Complete!")
    print("Results and visualizations saved in the 'results' folder")
    
    # Example prediction with new data
    print("\nExample prediction with sample data:")
    sample_data = X.iloc[:5]  # Take 5 samples from the dataset
    sample_actual = y.iloc[:5].values
    
    try:
        sample_predictions = predict_strike_efficiency(sample_data)
        
        # Compare predictions with actual values
        print("\nPrediction Results:")
        for i, (pred, actual) in enumerate(zip(sample_predictions, sample_actual)):
            print(f"Soldier {i+1}: Predicted={pred:.4f}, Actual={actual:.4f}, Difference={abs(pred-actual):.4f}")
        
        print("\nPrediction test successful!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}") 