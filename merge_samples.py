import rioxarray
import numpy as np
import xarray as xr
import os
from glob import glob
import matplotlib.pyplot as plt
import geopandas as gpd

year = 2021
parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
crop_map_file = f'{parent_dir}/CropMapOfEngland{year}-FGDB.fgb'


sample_tiles = ['30UUA', '30UUB', '30UVA',
                '30UVB', '30UVC', '30UVD', '30UVE', '30UVF', '30UVG', '30UWA',
                '30UWB', '30UWC', '30UWD', '30UWE', '30UWF', '30UWG', '30UXB',
                '30UXC', '30UXD', '30UXE', '30UXF',  '30UYB', '30UYC',
                '30UYD', '30UYE', '31UDT', '31UDU']

# crop_sample_temp = f'{parent_dir}/data/S2_*_monthly_comosite_{year}_crop_samples.nc'
# crop_samples = glob(crop_sample_temp)

# # for crop_sample in crop_samples:

lucodes = []
band_value_arrs = []
gdfs = []
for s2_tile in sample_tiles:
    
    crop_sample = f'{parent_dir}/data/S2_{s2_tile}_monthly_comosite_{year}_crop_samples.nc'
    
    gdf = gpd.read_file(crop_sample.replace('.nc', '.fgb'))
    
    print(f'Processing {crop_sample}...')
    
    ds = xr.open_dataset(crop_sample)
    lucode = ds['lucode'].values
    
    band_values = ds['band_values'].values
    bad_values = (band_values == 0).all(axis=(0,1))
    band_values = band_values[:, :, ~bad_values]
    
    lucode = lucode[~bad_values]
    gdf = gdf[~bad_values]

    gdfs.append(gdf)
    lucodes.append(lucode)
    band_value_arrs.append(band_values)

band_values = np.concatenate(band_value_arrs, axis=2)
lucode = np.concatenate(lucodes)
import pandas as pd
gdf = gpd.GeoDataFrame(pd.concat(gdfs))

# save as a new dataset
band_values = xr.DataArray(band_values, dims=['band', 'month', 'sample'],
                            coords={'band': ds['band_values'].band.values,
                                    'month': np.arange(1, 13),
                                    'sample': np.arange(band_values.shape[2])}
                            )
lucode = xr.DataArray(lucode, dims=['sample'], coords={'sample': np.arange(band_values.shape[2])})

band_values = xr.Dataset({'band_values': band_values, 'lucode': lucode})

band_values.to_netcdf(f'{parent_dir}/data/S2_{year}_crop_samples.nc')
gdf.to_file(f'{parent_dir}/data/S2_{year}_crop_samples.fgb', driver="FlatGeobuf")


    # wheat_mask = (lucode == 'AC66') 
    # non_wheat_mask = (lucode != 'AC66')
    
    # wheat = band_values[:, :, wheat_mask]
    # non_wheat = band_values[:, :, ~wheat_mask]
    
    # # ndvi = (wheat[7] - wheat[2]) / (wheat[7] + wheat[2])
    # # _ = plt.plot(ndvi, label='NDVI', lw=1, alpha=0.5)
    
    # # train a model to classify wheat and non-wheat
    # # then use the model to classify the entire dataset

    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score

    # X = band_values.reshape(-1, band_values.shape[2]).T
    # Y = np.zeros(X.shape[0])
    # Y[wheat_mask] = 1

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    # accuracy_score(Y_test, Y_pred)
    
    # break