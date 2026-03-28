import ee
import os
import xee
import numpy as np

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='gee-marc')


def generate_monthly_composites(s2_tile, year):
    # # Load the FeatureCollection for the specified year
    # table = ee.FeatureCollection(f'users/marcyinfeng/UK_CROP_MAP/CropMapOfEngland{year}')
    
    # # Filter for winter wheat
    # wheat = table.filter(ee.Filter.eq('lucode', 'AC66'))

    # Sentinel-2 Harmonized Image Collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Cloud Score+ Image Collection
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

    # Cloud score band and clear threshold
    QA_BAND = 'cs_cdf'
    CLEAR_THRESHOLD = 0.65

    tile_filter = ee.Filter.eq('MGRS_TILE', s2_tile)

    # Function to mask clouds using Cloud Score+
    def mask_clouds(img):
        return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

    # Filter, add cloud score bands, and mask clouds
    s2_filtered = s2.filter(tile_filter).filterDate(f'{year}-01-01', f'{year}-12-31') \
                    .map(lambda img: img.addBands(csPlus.filter(ee.Filter.eq('system:index', img.get('system:index'))).first())) \
                    .map(mask_clouds)

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
    

    # months = ee.List.sequence(1, 12)
    # image_list = months.map(do_monthly_mosaic)
    # bandTypes = dict(zip(sel_bands, ['uint16']*len(sel_bands)))
    # composite = ee.ImageCollection(image_list).cast(bandTypes, sel_bands)#.map(lambda image: image.toUint16())
    # # print(composite.getInfo())

    crs= s2.first().select('B2').projection().crs().getInfo()
    region = s2_filtered.geometry().dissolve()

    for month in range(1, 13):
        composite_month = do_monthly_mosaic(month)
        # clamp the image max value to 65535 and min value to 0
        composite_month = composite_month.clamp(0, 65535)
        composite_month = composite_month.uint16()
        asset_name = f'UK_Crop_Map_S2_{s2_tile}_{year}_{month}'
        # composite_month = composite.filter(ee.Filter.calendarRange(month, month+1, 'month')).first()
        task = ee.batch.Export.image.toAsset(
            image=composite_month,
            description=asset_name,
            assetId=f'users/marcyinfeng/UK_crop_map_S2_time_series/{asset_name}',
            region=region,
            scale=10,
            crs=crs,
            maxPixels=1e13
        )
        task.start()
        print(f'Exporting {asset_name} to asset...')


if __name__ == '__main__':
    s2_tile = '30UWB'
    year = 2020
    generate_monthly_composites(s2_tile, year)