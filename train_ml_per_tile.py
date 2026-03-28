# import os
# import numpy as np
# import pandas as pd
# import xarray as xr
# import xgboost as xgb
# import geopandas as gpd
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# year = 2018


# sample_tiles = ['30UUA', '30UUB', '30UVA',
#                 '30UVB', '30UVC', '30UVD', '30UVE', '30UVF', '30UVG', '30UWA',
#                 '30UWB', '30UWC', '30UWD', '30UWE', '30UWF', '30UWG', '30UXB',
#                 '30UXC', '30UXD', '30UXE', '30UXF',  '30UYB', '30UYC',
#                 '30UYD', '30UYE', '31UDT', '31UDU']

# parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'

# # crop_sample_temp = f'{parent_dir}/data/S2_*_monthly_comosite_{year}_crop_samples.nc'
# # crop_samples = glob(crop_sample_temp)

# # # for crop_sample in crop_samples:

# lucodes = []
# band_value_arrs = []
# gdfs = []
# for s2_tile in sample_tiles:
    
#     crop_sample = f'{parent_dir}/data/S2_{s2_tile}_monthly_comosite_{year}_crop_samples.nc'
    
#     gdf = gpd.read_file(crop_sample.replace('.nc', '.fgb'))
    
#     print(f'Processing {crop_sample}...')
    
#     ds = xr.open_dataset(crop_sample)
#     lucode = ds['lucode'].values

#     ds = xr.open_dataset(crop_sample)
#     lucode = ds['lucode'].values
    
#     band_values = ds['band_values'].values
#     bad_values = (band_values == 0).all(axis=(0,1))
#     band_values = band_values[:, :, ~bad_values]
    
#     lucode = lucode[~bad_values]
#     gdf = gdf[~bad_values]

#     band_values = band_values.reshape(-1, lucode.size)

#     X = pd.DataFrame(band_values.T, columns=[f'B{i}' for i in range(1, band_values.shape[0] + 1)])
#     y = lucode


# def train_model(X, y, year, tile):
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#     model = XGBClassifier(
#                         learning_rate =0.1,
#                         n_estimators=1000,
#                         max_depth=5,
#                         min_child_weight=1,
#                         gamma=0,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         # objective= 'binary:logistic',
#                         # nthread=4,
#                         # scale_pos_weight=1,
#                         seed=27,
#                         n_jobs=-1,
#                         tree_method="hist",
#                         verbosity=3,
#                         )


#     # Fit the model to the training data
#     model.fit(X_train, y_train, verbose=True)

#     # compute the accuracy of the model
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     # compute the confusion matrix

#     conf_matrix = confusion_matrix(y_test, y_pred)
#     print(conf_matrix)
#     # compute the classification report
#     class_report = classification_report(y_test, y_pred)
#     print(class_report)
#     # save the accuracy, confusion matrix, and classification report
#     with open(f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_{tile}_metrics.txt', 'w') as f:
#         f.write(f"Accuracy: {accuracy}\n")
#         f.write(f"Confusion Matrix:\n{conf_matrix}\n")
#         f.write(f"Classification Report:\n{class_report}\n")

#     model_path = f'/home/users/marcyin/marcyin/UK_crop_map/data/xgb_{year}_{tile}_model.json'
#     if os.path.exists(model_path):
#         os.remove(model_path)
#     model.save_model(model_path)
   


import os
import cudf
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import geopandas as gpd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define constants
YEAR = 2018
PARENT_DIR = '/home/users/marcyin/marcyin/UK_crop_map' 
OUTPUT_DIR = os.path.join(PARENT_DIR, "data")
SAMPLE_TILES = [
    '30UUA', '30UUB', '30UVA', '30UVB', '30UVC', '30UVD', '30UVE', '30UVF', '30UVG', 
    '30UWA', '30UWB', '30UWC', '30UWD', '30UWE', '30UWF', '30UWG', '30UXB', '30UXC', 
    '30UXD', '30UXE', '30UXF', '30UYB', '30UYC', '30UYD', '30UYE', '31UDT', '31UDU'
]

def load_data(tile, year, parent_dir):
    """
    Load data for a specific tile and year.
    """
    crop_sample_path = os.path.join(parent_dir, f"data/S2_{tile}_monthly_comosite_{year}_crop_samples.nc")
    metadata_path = crop_sample_path.replace(".nc", ".fgb")
    
    if not os.path.exists(crop_sample_path) or not os.path.exists(metadata_path):
        print(f"Missing data for tile {tile}. Skipping.")
        return None, None, None
    
    try:
        # Load geodataframe
        gdf = gpd.read_file(metadata_path)
        
        # Load band data
        ds = xr.open_dataset(crop_sample_path)
        lucode = ds['lucode'].values
        band_values = ds['band_values'].values
        
        # Filter out bad values
        bad_values = (band_values == 0).all(axis=(0, 1))
        band_values = band_values[:, :, ~bad_values]
        lucode = lucode[~bad_values]
        gdf = gdf[~bad_values]

        # Flatten bands for ML input
        band_values = band_values.reshape(-1, lucode.size)
        X = pd.DataFrame(band_values.T, columns=[f'B{i}' for i in range(1, band_values.shape[0] + 1)])
        y = lucode == 'AC66'
        if y.sum() == 0 or (~y).sum() == 0:
            print(f"No winter wheat or non-winter wheat samples for tile {tile}. Skipping.")
            return None, None, None
        return X, y, gdf
    
    except Exception as e:
        print(f"Error processing tile {tile}: {e}")
        return None, None, None


def train_and_save_model(X, y, year, tile, output_dir):
    """
    Train an XGBoost classifier and save the model and metrics.
    """
    if X is None or y is None or X.empty or y.size == 0:
        print(f"No data for tile {tile}. Skipping training.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    print(f"Training shapes for tile {tile}: X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Training shapes for tile {tile}: y_train: {y_train.shape}, y_test: {y_test.shape}")

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



    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Tile {tile} - Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f"xgb_{year}_{tile}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Classification Report:\n{class_report}\n")

    # Save model
    model_path = os.path.join(output_dir, f"xgb_{year}_{tile}_model.json")
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save_model(model_path)
    print(f"Model and metrics saved for tile {tile}.")


def main():
    for tile in SAMPLE_TILES:
        print(f"Processing tile {tile}...")
        X, y, gdf = load_data(tile, YEAR, PARENT_DIR)
        train_and_save_model(X, y, YEAR, tile, OUTPUT_DIR)


if __name__ == "__main__":
    main()
