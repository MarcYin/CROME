import xgboost as xgb
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def predict_zarr(zarr_path, model, bands, chunk_size):
    logging.info(f'Starting processing for {zarr_path}...')
    # try:
    ds = xr.open_zarr(zarr_path)
    ds.rio.set_spatial_dims('lon', 'lat', inplace=True)

    # Derive the output path from the Zarr path
    output_path = zarr_path.replace('.zarr', '_predictions.nc')

    # Prepare output array
    pred_array = np.zeros((ds.sizes['lon'], ds.sizes['lat']), dtype='float32')

    # Iterate through the dataset in chunks
    for lat_start in range(0, ds.sizes['lat'], chunk_size):
        for lon_start in range(0, ds.sizes['lon'], chunk_size):
            logging.info(f'Processing chunk: lat={lat_start}-{lat_start+chunk_size}, lon={lon_start}-{lon_start+chunk_size}')
            lat_end = min(lat_start + chunk_size, ds.sizes['lat'])
            lon_end = min(lon_start + chunk_size, ds.sizes['lon'])

            # Extract chunk
            subset = ds[bands].isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).load()

            subset = subset.where(np.isfinite(subset), 0).astype('uint16')
            subset = subset.to_dataarray()
            
            # Prepare data for model
            data = subset.values.reshape(subset.shape[0] * subset.shape[1], -1).T
            data = pd.DataFrame(data, columns=[f'B{i}' for i in range(1, subset.shape[0] * subset.shape[1] + 1)])
            dmatrix = xgb.DMatrix(data)

            # Predict
            pred = model.predict(dmatrix)
            pred = pred.reshape(subset.shape[2], subset.shape[3])

            # Store predictions in the output array
            pred_array[lon_start:lon_end, lat_start:lat_end] = pred
    # Save the prediction as a DataArray
    pred_da = xr.DataArray(
        pred_array,
        dims=['longitude', 'latitude'],
        coords={'longitude': ds.lon.values, 'latitude': ds.lat.values},
        name='prediction'
    )

    # Set spatial reference and save to file
    pred_da.rio.write_crs(ds.rio.crs, inplace=True)
    pred_da.rio.set_spatial_dims('longitude', 'latitude', inplace=True)
    pred_da.rio.write_coordinate_system(inplace=True)

    # pred_da.to_netcdf(output_path)
    # save it as geotiff
    pred_da = pred_da.transpose('latitude', 'longitude').astype('float32')
    
    
    pred_da.rio.to_raster(output_path.replace('.nc', '_tile_model.tif'), driver="GTiff", compress="LZW", tiled=True)

    logging.info(f'Successfully saved predictions to {output_path}')
    # except Exception as e:
    #     logging.error(f'Error processing {zarr_path}: {e}')



if __name__ == '__main__':
    # Configuration
    year = 2018
    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'

    tiles = ['29UQR', '29UQS', '30UUA', '30UUB', '30UVA',
            '30UVB', '30UVC', '30UVD', '30UVE', '30UVF', '30UVG', '30UWA',
            '30UWB', '30UWC', '30UWD', '30UWE', '30UWF', '30UWG', '30UXB',
            '30UXC', '30UXD', '30UXE', '30UXF', '30UXG', '30UYB', '30UYC',
            '30UYD', '30UYE', '31UCA', '31UCS', '31UCT', '31UCU', '31UCV',
            '31UDT', '31UDU']


    # Get tile index from command-line arguments
    try:
        tile_index = int(sys.argv[1]) - 1
        tile = tiles[tile_index]
    except IndexError:
        logging.error("Tile index out of range. Please provide a valid tile index.")
        sys.exit(1)
    except ValueError:
        logging.error("Invalid input. Please provide a numeric tile index.")
        sys.exit(1)

    zarr_path = f'{parent_dir}/data/S2_{tile}_monthly_comosite_{year}.zarr'
    model_path = f'{parent_dir}/data/xgb_{year}_{tile}_model.json'

    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    chunk_size = 1024  # Number of pixels to process along each dimension

    # Load the model
    try:
        model = xgb.Booster()
        model.load_model(model_path)
    except Exception as e:
        logging.error(f'Failed to load model from {model_path}: {e}')
        sys.exit(1)

    # Predict on the selected tile
    predict_zarr(zarr_path, model, bands, chunk_size)

