import dask_gateway

# Create a connection to dask-gateway.
gw = dask_gateway.Gateway("https://dask-gateway.jasmin.ac.uk", auth="jupyterhub")

# Inspect and change the options if required before creating your cluster.
options = gw.cluster_options()
options.worker_cores = 2
options.account = "nceo_generic"
# options.worker_memory = 20
# Create a Dask cluster, or, if one already exists, connect to it.
# This stage creates the scheduler job in Slurm, so it may take some
# time while your job queues.
clusters = gw.list_clusters()
if not clusters:
    cluster = gw.new_cluster(options, shutdown_on_close=False)
else:
    cluster = gw.connect(clusters[0].name)

# Create at least one worker, and allow your cluster to scale to three.
cluster.adapt(minimum=1, maximum=3)


# Get a Dask client.
client = cluster.get_client()
client

# print client status url
print(f"Client status url: {client.dashboard_link}")

# parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
# zarr_path = f'{parent_dir}/data/S2_31UDU_monthly_comosite_2018.zarr'



import os
import rioxarray
import xarray as xr
import numpy as np
import geopandas as gpd
import pyogrio
from tqdm import tqdm
from dask.diagnostics import ProgressBar


def extract_crop_samples(parent_dir, zarr_path, crop_map_file):
    """
    Extract crop samples from satellite imagery based on a crop map.
    
    Parameters:
    parent_dir (str): Path to the parent directory containing the data.
    zarr_path (str): Path to the Zarr file with the satellite imagery.
    crop_map_file (str): Path to the crop map file (FileGeoDatabase).
    
    Returns:
    geopandas.GeoDataFrame: GeoDataFrame containing the crop samples.
    """
    # Load the satellite imagery
    ds = xr.open_zarr(zarr_path)
    ds.rio.set_spatial_dims('lon', 'lat', inplace=True)

    bounds = ds.rio.transform_bounds(pyogrio.read_info(crop_map_file)['crs'])

    gdf = gpd.read_file(crop_map_file, columns=['lucode', 'geometry'], engine="pyogrio", bbox=bounds)
    gdf = gdf.to_crs(ds.rio.crs)
    
    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values
    
    # calculate the index of the nearest pixel from the geotransform
    geo_transform = ds.rio.transform().to_gdal()
    # transform the coordinates to pixel index
    x_idx = ((lons - geo_transform[0]) / geo_transform[1]).astype(int)
    y_idx = ((lats - geo_transform[3]) / geo_transform[5]).astype(int)


    # x_idx = np.clip(x_idx, 0, ds.sizes['lon'] - 1)
    # y_idx = np.clip(y_idx, 0, ds.sizes['lat'] - 1)

    # x_indx_nna = np.argmin(np.abs(lons[:1000] - ds.lon.values[:, np.newaxis]), axis=0)
    # y_indx_nna = np.argmin(np.abs(lats[:1000] - ds.lat.values[:, np.newaxis]), axis=0)
    # import pylab as plt
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].plot(x_idx[:1000], x_indx_nna, 'o', ms=2)
    # axs[0].plot([0, x_indx_nna.max()], [0, x_indx_nna.max()], 'r--', lw=1)
    # axs[1].plot(y_idx[:1000], y_indx_nna, 'o', ms=2)
    # axs[1].plot([0, y_indx_nna.max()], [0, y_indx_nna.max()], 'r--', lw=1)


    mask = (x_idx >= 0) & (x_idx < ds.sizes['lon']) & (y_idx >= 0) & (y_idx < ds.sizes['lat'])
    x_idx = x_idx[mask]
    y_idx = y_idx[mask]
    lucode = gdf.lucode.values[mask].astype(str)

    # lucodes = ['AC00', 'AC01', 'AC03', 'AC04', 'AC05', 'AC06', 'AC07', 'AC09',
    #             'AC10', 'AC100', 'AC14', 'AC15', 'AC16', 'AC17', 'AC18', 'AC19',
    #             'AC20', 'AC22', 'AC23', 'AC24', 'AC26', 'AC27', 'AC30', 'AC32',
    #             'AC34', 'AC35', 'AC36', 'AC37', 'AC38', 'AC41', 'AC44', 'AC45',
    #             'AC50', 'AC52', 'AC58', 'AC59', 'AC60', 'AC61', 'AC62', 'AC63',
    #             'AC64', 'AC65', 'AC66', 'AC67', 'AC68', 'AC69', 'AC70', 'AC71',
    #             'AC72', 'AC74', 'AC81', 'AC88', 'AC90', 'AC92', 'AC94', 'CA02',
    #             'FA01', 'HE02', 'HEAT', 'LG01', 'LG02', 'LG03', 'LG04', 'LG06',
    #             'LG07', 'LG08', 'LG09', 'LG11', 'LG13', 'LG14', 'LG15', 'LG16',
    #             'LG20', 'LG21', 'NA01', 'NU01', 'PG01', 'SR01', 'TC01', 'WA00',
    #             'WA01', 'WO12']

    # # check if all the lucodes are in the list
    # lucode_not_in = [i for i in lucode if i not in lucodes]
    # if lucode_not_in:
    #     raise ValueError(f'Lucodes {lucode_not_in} not in the list of lucodes!')

    # lucode_index = np.array([lucodes.index(l) for l in lucode])

    band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    with ProgressBar():
        band_vals = []
        for band_name in band_names:
            print(f'Processing {band_name}...')
            da = ds[band_name]
            da = da.where(np.isfinite(da), 0).astype('uint16')
            # now load the data into memory
            da = da.load()

            print(da.values[:, x_idx, y_idx].shape)
            # print(da.values[:, x_idx, y_idx])
            # # extract the values

            band_vals.append(da.values[:, x_idx, y_idx])

    # create one xarray dataarray for the band values with the coordindates be the band name, month(12) and the sample lucode
    band_vals = np.stack(band_vals, axis=0)
    band_vals = xr.DataArray(band_vals, dims=['band', 'month', 'sample'], 
                                        coords={'band': band_names, 
                                                 'month': np.arange(1, 13), 
                                                 'sample': np.arange(band_vals.shape[2])}
                            )
    # lucode_index = xr.DataArray(lucode_index, dims=['sample'], coords={'sample': np.arange(band_vals.shape[2])})
    
    lucode = xr.DataArray(lucode, dims=['sample'], coords={'sample': np.arange(band_vals.shape[2])})

    # create a dataset
    band_vals = xr.Dataset({'band_values': band_vals, 'lucode': lucode})
    
    print(band_vals)
    # save the dataarray
    band_vals.to_netcdf(zarr_path.replace('.zarr', '_crop_samples.nc'))
    # save the subset gdf
    gdf = gdf[mask]
    gdf.to_file(zarr_path.replace('.zarr', '_crop_samples.fgb'), driver="FlatGeobuf")
    
if __name__ == '__main__':
    year = 2022
    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map'
    crop_map_file = f'{parent_dir}/CropMapOfEngland{year}-FGDB.fgb'

    zarr_path_temp = f'{parent_dir}/data/S2_*_monthly_comosite_{year}.zarr'
    from glob import glob
    zarr_paths = glob(zarr_path_temp)

    for zarr_path in zarr_paths:
        print(f'Processing {zarr_path}...')
        crop_sample_file = zarr_path.replace('.zarr', '_crop_samples.fgb')
        if not os.path.exists(crop_sample_file):
            extract_crop_samples(parent_dir, zarr_path, crop_map_file)
        else:
            print(f'{crop_sample_file} already exists!')

# Shut down the cluster
cluster.shutdown()








# import rioxarray
# import xarray as xr
# import numpy as np
# import geopandas as gpd
# import pyogrio

# crop_map_file = f'{parent_dir}/CropMapOfEngland2018-FGDB.fgb'
# crop_map_info = pyogrio.read_info(crop_map_file)


# ds = xr.open_zarr(zarr_path)
# B2 = ds.B2
# B2.rio.set_spatial_dims('lon', 'lat', inplace=True)
# bounds = B2.rio.transform_bounds(crop_map_info['crs'])

# target_columns = ['lucode', 'geometry']
# gdf = gpd.read_file(crop_map_file, bbox=bounds, columns=target_columns, engine="pyogrio")

# gdf = gdf.to_crs(B2.rio.crs)

# gdf['x'] = gdf.geometry.x
# gdf['y'] = gdf.geometry.y
# samples = gdf[['x', 'y']].values


# # Use a context manager to ensure the client is properly closed
# with client:
#     for band_name in ds.data_vars:
#         print(f'Processing {band_name}...')

#         da = ds[band_name]
#         da = da.where(np.isfinite(da), 0).astype('uint16')
        
#         # now we can use the x and y to get the values from the raster
#         values = da.sel(lon=samples[:, 0], lat=samples[:, 1], method='nearest').values
#         gdf[band_name] = values.compute()

# gdf.to_file(zarr_path.replace('.zarr', '_crop_samples.fgb'), driver="FlatGeobuf")


# # Shut down the cluster
# cluster.shutdown()


# output_raster = zarr_path.replace('.zarr', f'_{band_name}.tif')
# print(f'Saving to {output_raster}...')
# # change coordinate name from lon to x and lat to y
# da = da.rename({'lon': 'x', 'lat': 'y'})
# da.rio.set_spatial_dims('x', 'y', inplace=True)
# da = da.transpose('time', 'y', 'x')
# da.rio.to_raster(output_raster, driver="GTiff", compress="LZW")





# # import rioxarray
# # import xarray as xr

# # ds = xr.open_dataset(zarr_path)
# # ds.B2[0].plot.imshow()

# import apache_beam as beam
# import numpy as np
# import pandas as pd
# import xarray_beam as xbeam
# import xarray
# from apache_beam.options.pipeline_options import PipelineOptions

# parent_dir = '/home/users/marcyin/marcyin/UK_crop_map/data'
# zarr_path = f'{parent_dir}/S2_31UDU_monthly_comosite_2018.zarr'

# ds_on_disk, chunks = xbeam.open_zarr(zarr_path)

# options = PipelineOptions(runner='DirectRunner',
#                             direct_num_workers = 4,
#                             direct_running_mode='multi_processing'
#                             )

# with beam.Pipeline(options=options) as p:
#     p | xbeam.DatasetToChunks(ds_on_disk, chunks) | beam.MapTuple(lambda k, v: print(k, type(v)))


# # store = zarr.DirectoryStore(zarr_path)
# # root = zarr.open(store)
# # print(root.tree())

# # # read the data
# # data = root['B2']
# # print(data.shape)
# # print(data[:, 1000, 1000])


# # import rioxarray
# # import xarray as xr

# # ds = xr.open_dataset(zarr_path)
# # ds.B2[:, 1000, 1000].plot.imshow()
