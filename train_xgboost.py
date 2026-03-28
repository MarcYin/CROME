import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cudf

# Function to load crop samples
def load_crop_samples(year):
    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
    crop_samples = f'{parent_dir}/data/S2_{year}_crop_samples.nc'
    ds = xr.open_dataset(crop_samples)

    lucode = ds['lucode'].values
    band_values = ds['band_values'].values.reshape(-1, lucode.size)

    y = lucode == 'AC66'  # Binary classification for 'AC66'
    le_name_mapping = {'AC66': 1, 'Other': 0}

    # Create column names
    band_name = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    month = np.arange(1, 13)
    column_names = [f'{b}_{m}' for b in band_name for m in month]

    df = pd.DataFrame(band_values.T, columns=column_names)
    df['lucode'] = y.astype(int)

    return df, le_name_mapping

# Function to train the model
def train_model(X_train, X_test, y_train, y_test, year):
    X_train = cudf.DataFrame.from_pandas(X_train)
    y_train = cudf.Series.from_pandas(y_train)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        device="cuda",
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=15,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        seed=27,
        n_jobs=-1,
        tree_method="hist",
        verbosity=3
    )

    # Train the model
    import time
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    print("GPU Training Time: %s seconds" % (str(time.time() - start)))

    # Save model
    model_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_model_winter_wheat.json'
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save_model(model_path)

    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, year):
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Save metrics
    metrics_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_metrics_cudf.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Classification Report:\n{class_report}\n")
    
    print(f"Metrics saved to {metrics_path}")
    return accuracy, conf_matrix, class_report

# Main script
if __name__ == "__main__":
    # # Parse year from command line arguments
    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py <year>")
    #     sys.exit(1)
    
    # try:
    #     year = int(sys.argv[1])
    # except ValueError:
    #     print("Year must be an integer.")
    #     sys.exit(1)
    for year in range(2018, 2021):
        print(f'Doing year: {year}')
        # Load data
        data, le_name_mapping = load_crop_samples(year)
        X = data.drop('lucode', axis=1)
        y = data['lucode']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

        # Train model
        model = train_model(X_train, X_test, y_train, y_test, year)

        # Evaluate model
        accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test, year)

        # Print evaluation metrics
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")
