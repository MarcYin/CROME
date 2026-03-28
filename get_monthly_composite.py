import ee
import os
import xee
import numpy as np
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
from apache_beam.runners.dask.dask_runner import DaskRunner
from dask.distributed import Client
from apache_beam.options.pipeline_options import PipelineOptions

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='gee-marc')


def generate_monthly_composites(s2_tile, year, parent_dir = './'):
    # # Load the FeatureCollection for the specified year
    # table = ee.FeatureCollection(f'users/marcyinfeng/UK_CROP_MAP/CropMapOfEngland{year}')
    
    # # Filter for winter wheat
    # wheat = table.filter(ee.Filter.eq('lucode', 'AC66'))

    # Sentinel-2 Harmonized Image Collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Cloud Score+ Image Collection
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

    # Cloud score band and clear threshold
    QA_BAND = 'cs'
    CLEAR_THRESHOLD = 0.65

    tile_filter = ee.Filter.eq('MGRS_TILE', s2_tile)

    # # Function to mask clouds using Cloud Score+
    # def mask_clouds(img):
    #     return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))


    # # Filter, add cloud score bands, and mask clouds
    # s2_filtered = s2.filter(tile_filter).filterDate(f'{year}-01-01', f'{year}-12-31') \
    #                 .map(lambda img: img.addBands(csPlus.filter(ee.Filter.eq('system:index', img.get('system:index'))).first())) \
    #                 .map(mask_clouds)

    s2_filtered = s2.filter(tile_filter).filterDate(f'{year}-01-01', f'{year}-12-31')\
                    .linkCollection(csPlus, [QA_BAND])\
                    .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))

    sel_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    # Function to create a monthly mosaic
    def do_monthly_mosaic(month):
        month_start = ee.Number(month)
        month_end = month_start.add(1)
        return s2_filtered.filter(ee.Filter.calendarRange(month_start, month_end, 'month')) \
                          .limit(15, 'CLOUDY_PIXEL_PERCENTAGE', True) \
                          .select(sel_bands) \
                          .median() \
                          .set('system:time_start', ee.Date.fromYMD(year, month, 1))
                        #   \
                        #   .cast({'B.*': 'uint16'})
    

    months = ee.List.sequence(1, 12)
    image_list = months.map(do_monthly_mosaic)
    bandTypes = dict(zip(sel_bands, ['uint16']*len(sel_bands)))
    composite = ee.ImageCollection(image_list).cast(bandTypes, sel_bands)#.map(lambda image: image.toUint16())
    # print(composite.getInfo())
    # crs= s2.first().select('B2').projection().crs().getInfo()

    region = s2_filtered.geometry().dissolve()
    # print(region.getInfo())
    ds = xr.open_dataset(
        composite,
        crs= 'EPSG:4326',
        geometry = region,
        # scale= 10,
        projection = ee.Projection('EPSG:4326').atScale(10),
        engine=xee.EarthEngineBackendEntrypoint,
    )
    # .isel(time=0).to_dataarray().rename({'X': 'x', 'Y': 'y'}).transpose('variable', 'y', 'x')
    time_coords = np.array([np.datetime64(str(year) + '-' + str(month).zfill(2)) for month in range(1, 13)])
    ds['time'] = time_coords.astype('datetime64[ns]')
    print(ds)
    # ds.rio.to_raster(filename)
    
    template = xbeam.make_template(ds)
    
    itemsize = max(variable.dtype.itemsize for variable in template.values())
    target_chunks = {'time': len(time_coords), 'lon': 512, 'lat': 512}
    # target_chunks = {'time': 8, 'Y': 512, 'X': 512}

    options = PipelineOptions(runner='DirectRunner',
                              direct_num_workers = 1,
                            #   direct_running_mode='multi_threading'
                              )
    fname = f'{parent_dir}/S2_{s2_tile}_monthly_comosite_{year}.zarr'
    print(f'Saving to {fname}...')
    with beam.Pipeline(options = options) as root:
        _ = (
            root
            | xbeam.DatasetToChunks(ds, target_chunks)
            # | beam.MapTuple(lambda k, v: (k, v.where(np.isfinite(v), 0).astype('uint16')))
            | xbeam.ChunksToZarr(fname, template, target_chunks)
        )


    # options = PipelineOptions(runner=DaskRunner(), direct_num_workers = 8)
    # with beam.Pipeline(options = options) as root:
    #     _ = (
    #         root
    #         | xbeam.DatasetToChunks(ds, target_chunks)
    #         # | beam.MapTuple(lambda k, v: (k, v.where(np.isfinite(v), 0).astype('uint16')))
    #         | xbeam.ChunksToZarr(f'{parent_dir}/S2_{s2_tile}_monthly_comosite_{year}.zarr', template, target_chunks)
    #     )

    # p = beam.Pipeline(options = options)
    # _ = (
    #     p
    #     | xbeam.DatasetToChunks(ds, source_chunks)
    #     | beam.MapTuple(lambda k, v: (k, v.where(np.isfinite(v), 0).astype('uint16')))
    #     | xbeam.ChunksToZarr(f'S2_{s2_tile}_monthly_comosite_{year}.zarr', template, target_chunks)
    # )


    # client = Client(n_workers=8)
    # with beam.Pipeline(
    #     runner=DaskRunner(),
    #     options=PipelineOptions(
    #             ["--dask_client_address", client.cluster.scheduler_address]
    #         )) as root:
    #     _ = (
    #         root
    #         | xbeam.DatasetToChunks(ds, source_chunks)
    #         | beam.MapTuple(lambda k, v: (k, v.where(np.isfinite(v), 0).astype('uint16')))
    #         | xbeam.ChunksToZarr(f'S2_{s2_tile}_monthly_comosite_{year}.zarr', template, target_chunks)
    #     )

    return ds


if __name__ == '__main__':


    import time
    import sys
    ind = int(sys.argv[1]) - 1
    # if (ind < 32) | (ind == 35):
    #     sys.exit()

    parent_dir = '/home/users/marcyin/marcyin/UK_crop_map/data'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    tiles = ['29UQR', '29UQS', '30UUA', '30UUB', '30UVA',
        '30UVB', '30UVC', '30UVD', '30UVE', '30UVF', '30UVG', '30UWA',
        '30UWB', '30UWC', '30UWD', '30UWE', '30UWF', '30UWG', '30UXB',
        '30UXC', '30UXD', '30UXE', '30UXF', '30UXG', '30UYB', '30UYC',
        '30UYD', '30UYE', '31UCA', '31UCS', '31UCT', '31UCU', '31UCV',
        '31UDT', '31UDU']
    
    tile = tiles[ind]
    year = 2023

    print(f'Generating monthly composites for {tile} in {year}...')
    start = time.time()
    generate_monthly_composites(tile, year, parent_dir)
    end = time.time()
    print(f'Time taken: {(end - start)/60} minutes')
    fname = f'{parent_dir}/S2_{tile}_monthly_comosite_{year}.zarr'
    # ds = xr.open_zarr(fname)
    # print(ds)
    # export image as geoTIFF



    # for tile in [ '30UXD', '30UYC', '30UXC']:
    #     for year in range(2018, 2024):


# # Example usage
# s2_tile = '30UYD'
# year = 2020
# import time
# start = time.time()
# ds = generate_monthly_composites(s2_tile, year)
# end = time.time()
# # total time in minutes
# print(f'Time taken: {(end - start)/60} minutes')
# # ds.to_netcdf('test.nc')
# # print(monthly_composites.getInfo())


# # ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
# # ds = xr.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=0.25)


# import xarray as xr
# parent_dir = '/home/users/marcyin/marcyin/UK_crop_map/data'
# da = xr.open_dataset(f'{parent_dir}/S2_30UXD_monthly_comosite_2018.zarr')
